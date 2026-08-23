from datetime import UTC, datetime, timedelta

from scraper import KfcScraper, _first_price


def test_first_price_accepts_baht_and_dot_dash_notation():
    assert _first_price("ราคา ฿179.00") == "฿179.00"
    assert _first_price("10 ชิ้น 199.-") == "฿199.00"
    assert _first_price("ราคา 99 บาท") == "฿99.00"


def test_contentful_menu_parser_extracts_thai_name_price_and_description():
    scraper = KfcScraper()
    entry = {
        "fields": {
            "name": "The Box Signature",
            "body": {
                "nodeType": "document",
                "content": [
                    {"nodeType": "text", "value": "เดอะบอกซ์ ซิกเนเจอร์"},
                    {"nodeType": "text", "value": "ไก่ทอด 2 ชิ้น"},
                    {"nodeType": "text", "value": "฿179.00"},
                ],
            },
        }
    }

    records = scraper._menu_from_contentful([entry], "2026-08-03T00:00:00Z")

    assert len(records) == 1
    assert records[0]["name"] == "เดอะบอกซ์ ซิกเนเจอร์"
    assert records[0]["price"] == "฿179.00"
    assert records[0]["description"] == "ไก่ทอด 2 ชิ้น"
    assert "The Box Signature" in records[0]["aliases"]


def test_supplemental_item_parser_maps_volcano_wing_to_its_thai_menu_name():
    scraper = KfcScraper()
    entries = [
        {
            "fields": {
                "itemName": "Volcano WingZ Gang (CAT41-592)",
                "longDescription": "",
            }
        },
        {
            "fields": {
                "itemName": "Volcano Wing (CAT41-446)",
                "longDescription": "วิงซ์เคลือบซอสฉ่ำๆ เผ็ดเข้มข้น อร่อยจนหยุดไม่ได้",
            }
        },
    ]

    records = scraper._supplemental_menu_from_contentful(entries, "2026-08-03T00:00:00Z")

    assert len(records) == 1
    assert records[0]["name"] == "วิงซ์ภูเขาไฟระเบิด"
    assert records[0]["price"] == ""
    assert "เผ็ดเข้มข้น" in records[0]["description"]
    assert "Volcano WingZ" in records[0]["aliases"]


def test_olo_menu_parser_reads_current_thai_catalog_contents_and_price():
    scraper = KfcScraper()
    catalog = {
        "categories": [
            {
                "products": [
                    {
                        "items": [
                            {
                                "id": "CAT41-613",
                                "name": "พอใจ บักเก็ต",
                                "dname": [
                                    {"lang": "th-TH", "value": "พอใจ บักเก็ต"}
                                ],
                                "longDescription": [
                                    {"lang": "th-TH", "value": "พอใจ บักเก็ต"}
                                ],
                                "modgrpIds": [
                                    {
                                        "name": "ไก่ทอด 4 ชิ้น",
                                        "modifiers": [{"name": "ไก่ทอด 4 ชิ้น"}],
                                    },
                                    {
                                        "name": "ชิคเก้น ป๊อป 7 ชิ้น",
                                        "modifiers": [{"name": "ชิคเก้น ป๊อป 7 ชิ้น"}],
                                    },
                                    {
                                        "name": "ไก่ทอด 1 ชิ้น",
                                        "modifiers": [
                                            {"name": "ไก่กรอบฮอทแอนด์สไปซี่"},
                                            {"name": "ไก่สูตรผู้พัน"},
                                        ],
                                    },
                                    {
                                        "name": "เลือกเครื่องเคียง",
                                        "modifiers": [
                                            {"name": "เฟรนช์ฟรายส์ (ปกติ)"},
                                            {"name": "มันบด"},
                                        ],
                                    },
                                ],
                                "availability": [{"price": 19900}],
                                "isHidden": False,
                                "isCategoryHidden": False,
                            }
                        ]
                    }
                ]
            }
        ]
    }

    records = scraper._menu_from_olo_catalog(catalog, "2026-08-03T00:00:00Z")

    assert len(records) == 1
    assert records[0]["name"] == "พอใจ บักเก็ต"
    assert records[0]["price"] == "฿199.00"
    assert records[0]["components"] == ["ไก่ทอด 4 ชิ้น", "ชิคเก้น ป๊อป 7 ชิ้น"]
    assert records[0]["choices"] == [
        {
            "group": "ไก่ทอด 1 ชิ้น",
            "options": ["ไก่กรอบฮอทแอนด์สไปซี่", "ไก่สูตรผู้พัน"],
        },
        {
            "group": "เลือกเครื่องเคียง",
            "options": ["เฟรนช์ฟรายส์ (ปกติ)", "มันบด"],
        },
    ]


def test_rendered_menu_image_parser_reads_lab_classes_and_strips_query():
    html = """
    <div class="menu-card">
      <div class="menu-product-header">เดอะบอกซ์ ซิกเนเจอร์</div>
      <img class="false medium-menu-product-image"
           src="https://images.example/kfc-box.png?width=640&quality=80" />
      <span>฿179.00</span>
    </div>
    """

    records = KfcScraper._image_records_from_html(
        html, "https://www.kfc.co.th/menu/meals", 10
    )

    assert records == [
        {
            "name": "เดอะบอกซ์ ซิกเนเจอร์",
            "image_url": "https://images.example/kfc-box.png",
            "price": "฿179.00",
            "description": "฿179.00",
            "url": "https://www.kfc.co.th/menu/meals",
        }
    ]


def test_promotion_parser_skips_expired_records_and_keeps_active_record():
    scraper = KfcScraper()
    now = datetime.now(UTC)
    expired = {
        "fields": {
            "headline": "โปรเก่า",
            "startDate": (now - timedelta(days=9)).isoformat(),
            "endDate": (now - timedelta(days=2)).isoformat(),
        }
    }
    active = {
        "fields": {
            "headline": "10 ชิ้น 199.-",
            "subHeadline": "ดีลวันอังคาร",
            "startDate": (now - timedelta(days=1)).isoformat(),
            "endDate": (now + timedelta(days=1)).isoformat(),
        }
    }

    records = scraper._promotion_from_contentful(
        [expired, active], "2026-08-03T00:00:00Z", dated=True
    )

    assert [record["name"] for record in records] == ["10 ชิ้น 199.-"]
    assert records[0]["price"] == "฿199.00"
