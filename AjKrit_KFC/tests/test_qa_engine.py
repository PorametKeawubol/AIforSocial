import json

from qa_engine import KfcQuestionAnswerer


def _write_snapshot(path):
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "items": [
                    {
                        "id": "box-signature",
                        "kind": "menu",
                        "name": "เดอะบอกซ์ ซิกเนเจอร์",
                        "price": "฿179.00",
                        "description": "ไก่ทอด 2 ชิ้น ไก่วิงซ์แซ่บ 2 ชิ้น ทาร์ตไข่ 1 ชิ้น",
                        "category": "เมนู KFC",
                        "aliases": ["The Box Signature"],
                        "url": "https://www.kfc.co.th/menu/meals",
                    },
                    {
                        "id": "wingz",
                        "kind": "menu",
                        "name": "ไก่วิงซ์แซ่บ 3 ชิ้น",
                        "price": "฿69.00",
                        "description": "ปีกไก่รสจัดจ้าน",
                        "category": "เมนู KFC",
                        "aliases": ["Wingz Zabb"],
                        "url": "https://www.kfc.co.th/menu/meals",
                    },
                    {
                        "id": "deal",
                        "kind": "promotion",
                        "name": "10 ชิ้น 199.-",
                        "price": "฿199.00",
                        "description": "ดีลวันอังคาร เฉพาะรับที่ร้าน",
                        "category": "โปรโมชัน KFC",
                        "aliases": ["ดีลวันอังคาร"],
                        "url": "https://www.kfc.co.th/promos-rewards",
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _add_volcano_wing(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["items"].append(
        {
            "id": "volcano-wingz",
            "kind": "menu",
            "name": "วิงซ์ภูเขาไฟระเบิด",
            "price": "",
            "description": "วิงซ์เคลือบซอสฉ่ำๆ เผ็ดเข้มข้น อร่อยจนหยุดไม่ได้",
            "category": "เมนู KFC",
            "aliases": ["Volcano WingZ", "Volcano Wing"],
            "url": "https://www.kfc.co.th/menu/meals",
        }
    )
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _add_porjai_bucket(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["items"].append(
        {
            "id": "porjai-bucket",
            "kind": "menu",
            "name": "พอใจ บักเก็ต",
            "price": "฿199.00",
            "description": (
                "ประกอบด้วย/เลือกได้: ไก่ทอด 4 ชิ้น, ชิคเก้น ป๊อป 7 ชิ้น, "
                "เลือกเครื่องเคียง, ดิปซอส"
            ),
            "category": "เมนู KFC",
            "aliases": ["Porjai Bucket"],
            "url": "https://www.kfc.co.th/menu/meals",
        }
    )
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _add_menu_with_selectable_chicken(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["items"].append(
        {
            "id": "choice-box",
            "kind": "menu",
            "name": "เดอะบอกซ์ เลือกไก่",
            "price": "฿159.00",
            "description": "เลือกชุดไก่ได้ตามต้องการ",
            "components": ["นักเก็ตส์ 2 ชิ้น"],
            "choices": [
                {
                    "group": "ไก่ทอด 1 ชิ้น",
                    "options": ["ไก่กรอบฮอทแอนด์สไปซี่", "ไก่สูตรผู้พัน"],
                },
                {
                    "group": "เลือกเครื่องเคียง",
                    "options": ["เฟรนช์ฟรายส์ (ปกติ)", "มันบด"],
                },
            ],
            "category": "เมนู KFC",
            "aliases": ["Choice Box"],
            "url": "https://www.kfc.co.th/menu/meals",
        }
    )
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _add_zinger_search_pair(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["items"].extend(
        [
            {
                "id": "zinger-burger",
                "kind": "menu",
                "name": "ซิงเกอร์ เบอร์เกอร์",
                "price": "฿89.00",
                "description": "เบอร์เกอร์ไก่ทอดซิงเกอร์รสจัดจ้าน",
                "category": "เมนู KFC",
                "aliases": ["Zinger Burger"],
                "url": "https://www.kfc.co.th/menu/meals",
            },
            {
                "id": "zinger-set",
                "kind": "menu",
                "name": "ชุดซิงเกอร์เบอร์เกอร์",
                "price": "฿129.00",
                "description": "ซิงเกอร์เบอร์เกอร์พร้อมเครื่องเคียง",
                "category": "เมนู KFC",
                "aliases": ["Zinger Burger Set"],
                "url": "https://www.kfc.co.th/menu/meals",
            },
        ]
    )
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _add_bucket_search_items(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["items"].extend(
        [
            {
                "id": "sukjai-bucket",
                "kind": "menu",
                "name": "สุขใจ บักเก็ต",
                "price": "฿449.00",
                "description": "ไก่ทอดและเครื่องเคียง",
                "category": "เมนู KFC",
                "aliases": ["สุขใจ บักเก็ต"],
                "url": "https://www.kfc.co.th/menu/meals",
            },
            {
                "id": "jujai-bucket",
                "kind": "menu",
                "name": "จุใจ บักเก็ต",
                "price": "฿349.00",
                "description": "ไก่ทอดและเครื่องเคียง",
                "category": "เมนู KFC",
                "aliases": ["จุใจ บักเก็ต"],
                "url": "https://www.kfc.co.th/menu/meals",
            },
            {
                "id": "porjai-bucket-search",
                "kind": "menu",
                "name": "พอใจ บักเก็ต",
                "price": "฿199.00",
                "description": "ไก่ทอดและเครื่องเคียง",
                "category": "เมนู KFC",
                "aliases": ["พอใจ บักเก็ต"],
                "url": "https://www.kfc.co.th/menu/meals",
            },
        ]
    )
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _add_wingz_search_items(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["items"].extend(
        [
            {
                "id": "wingz-two",
                "kind": "menu",
                "name": "วิงซ์แซ่บ 2 ชิ้น",
                "price": "฿49.00",
                "description": "ปีกไก่ทอดรสแซ่บ",
                "components": ["วิงซ์แซ่บ 2 ชิ้น"],
                "category": "เมนู KFC",
                "aliases": ["WingZ 2 pieces"],
                "url": "https://www.kfc.co.th/menu/meals",
            },
            {
                "id": "wingz-set",
                "kind": "menu",
                "name": "ชุดไก่วิงซ์แซ่บ 2 ชิ้น",
                "price": "฿99.00",
                "description": "ชุดวิงซ์แซ่บพร้อมเครื่องเคียง",
                "components": ["วิงซ์แซ่บ 2 ชิ้น", "เฟรนช์ฟรายส์"],
                "category": "เมนู KFC",
                "aliases": ["WingZ set"],
                "url": "https://www.kfc.co.th/menu/meals",
            },
            {
                "id": "bucket-with-wingz",
                "kind": "menu",
                "name": "บักเก็ตสุดคุ้ม",
                "price": "฿199.00",
                "description": "ชุดไก่ทอดรวม",
                "components": ["วิงซ์แซ่บ 4 ชิ้น"],
                "category": "เมนู KFC",
                "aliases": ["Value bucket"],
                "url": "https://www.kfc.co.th/menu/meals",
            },
        ]
    )
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_answers_an_exact_menu_question_without_loading_bert(tmp_path):
    data_file = tmp_path / "menu.json"
    _write_snapshot(data_file)
    answerer = KfcQuestionAnswerer(
        data_file=data_file, index_file=tmp_path / "index.npz", semantic_enabled=False
    )

    reply = answerer.answer("เดอะบอกซ์ ซิกเนเจอร์ คืออะไร")

    assert "เดอะบอกซ์ ซิกเนเจอร์" in reply
    assert "฿179.00" in reply
    assert "ไก่ทอด 2 ชิ้น" in reply


def test_lists_promotions_for_a_general_promotion_question(tmp_path):
    data_file = tmp_path / "menu.json"
    _write_snapshot(data_file)
    answerer = KfcQuestionAnswerer(
        data_file=data_file, index_file=tmp_path / "index.npz", semantic_enabled=False
    )

    reply = answerer.answer("มีโปรโมชั่นอะไรบ้าง")

    assert "โปรโมชัน KFC" in reply
    assert "10 ชิ้น 199.-" in reply


def test_broad_chicken_question_reports_total_and_all_related_items(tmp_path):
    data_file = tmp_path / "menu.json"
    _write_snapshot(data_file)
    answerer = KfcQuestionAnswerer(
        data_file=data_file, index_file=tmp_path / "index.npz", semantic_enabled=False
    )

    reply = answerer.answer("มีเมนูไก่ทอดอะไรบ้าง")

    assert "เมนู KFC ทั้งหมด 2 เมนู" in reply
    assert "พบ 2 เมนูที่เกี่ยวกับ “ไก่ทอด”" in reply
    assert "1. เดอะบอกซ์ ซิกเนเจอร์" in reply
    assert "2. ไก่วิงซ์แซ่บ 3 ชิ้น" in reply
    assert "ราคา" not in reply


def test_general_menu_question_reports_total_and_all_items(tmp_path):
    data_file = tmp_path / "menu.json"
    _write_snapshot(data_file)
    answerer = KfcQuestionAnswerer(
        data_file=data_file, index_file=tmp_path / "index.npz", semantic_enabled=False
    )

    reply = answerer.answer("มีเมนูอะไรบ้าง")

    assert "พบ 2 เมนู KFC" in reply
    assert "เดอะบอกซ์ ซิกเนเจอร์" in reply
    assert "ไก่วิงซ์แซ่บ 3 ชิ้น" in reply


def test_a_named_menu_wins_over_a_broad_question_phrase(tmp_path):
    data_file = tmp_path / "menu.json"
    _write_snapshot(data_file)
    answerer = KfcQuestionAnswerer(
        data_file=data_file, index_file=tmp_path / "index.npz", semantic_enabled=False
    )

    reply = answerer.answer("เมนูนี้มีอะไรบ้าง เดอะบอกซ์ ซิกเนเจอร์")

    assert "เดอะบอกซ์ ซิกเนเจอร์" in reply
    assert "ไก่วิงซ์แซ่บ 2 ชิ้น" in reply
    assert "ไก่วิงซ์แซ่บ 3 ชิ้น" not in reply


def test_detail_prompt_uses_the_full_name_after_the_question(tmp_path):
    data_file = tmp_path / "menu.json"
    _write_snapshot(data_file)
    _add_volcano_wing(data_file)
    answerer = KfcQuestionAnswerer(
        data_file=data_file, index_file=tmp_path / "index.npz", semantic_enabled=False
    )

    reply = answerer.answer("เมนูนี้มีอะไรบ้าง วิงซ์ภูเขาไฟระเบิด")

    assert "วิงซ์ภูเขาไฟระเบิด" in reply
    assert "เผ็ดเข้มข้น" in reply
    assert "ยังไม่พบเมนู" not in reply


def test_detail_prompt_finds_current_porjai_bucket_and_its_contents(tmp_path):
    data_file = tmp_path / "menu.json"
    _write_snapshot(data_file)
    _add_porjai_bucket(data_file)
    answerer = KfcQuestionAnswerer(
        data_file=data_file, index_file=tmp_path / "index.npz", semantic_enabled=False
    )

    reply = answerer.answer("เมนูนี้มีอะไรบ้าง พอใจ บักเก็ต")

    assert "พอใจ บักเก็ต" in reply
    assert "฿199.00" in reply
    assert "ไก่ทอด 4 ชิ้น" in reply
    assert "ชิคเก้น ป๊อป 7 ชิ้น" in reply
    assert "ยังไม่พบเมนู" not in reply


def test_detail_reply_lists_each_selectable_option(tmp_path):
    data_file = tmp_path / "menu.json"
    _write_snapshot(data_file)
    _add_menu_with_selectable_chicken(data_file)
    answerer = KfcQuestionAnswerer(
        data_file=data_file, index_file=tmp_path / "index.npz", semantic_enabled=False
    )

    reply = answerer.answer("เมนูนี้มีอะไรบ้าง เดอะบอกซ์ เลือกไก่")

    assert "📦 ประกอบด้วย:" in reply
    assert "🎛️ เลือกได้:" in reply
    assert "• ไก่ทอด 1 ชิ้น" in reply
    assert "  - ไก่กรอบฮอทแอนด์สไปซี่" in reply
    assert "  - ไก่สูตรผู้พัน" in reply
    assert "• เลือกเครื่องเคียง" in reply


def test_detail_prompt_reports_a_missing_name_instead_of_a_broad_list(tmp_path):
    data_file = tmp_path / "menu.json"
    _write_snapshot(data_file)
    answerer = KfcQuestionAnswerer(
        data_file=data_file, index_file=tmp_path / "index.npz", semantic_enabled=False
    )

    reply = answerer.answer("เมนูนี้มีอะไรบ้าง เมนูที่ไม่มีอยู่จริง")

    assert reply == "ไม่พบเมนูชื่อ “เมนูที่ไม่มีอยู่จริง” ในข้อมูลล่าสุด"


def test_help_is_available_without_a_data_file(tmp_path):
    answerer = KfcQuestionAnswerer(
        data_file=tmp_path / "absent.json",
        index_file=tmp_path / "index.npz",
        semantic_enabled=False,
    )

    assert "ตัวอย่าง" in answerer.answer("help")


def test_menu_search_treats_menu_synonyms_as_a_two_item_overview(tmp_path):
    data_file = tmp_path / "menu.json"
    _write_snapshot(data_file)
    answerer = KfcQuestionAnswerer(
        data_file=data_file, index_file=tmp_path / "index.npz", semantic_enabled=False
    )

    results = answerer.search_items("เมนูอาหาร", kind="menu", limit=2)

    assert len(results) == 2
    assert [item["name"] for _, item in results] == [
        "เดอะบอกซ์ ซิกเนเจอร์",
        "ไก่วิงซ์แซ่บ 3 ชิ้น",
    ]


def test_menu_search_finds_a_misspelled_partial_product_name(tmp_path):
    data_file = tmp_path / "menu.json"
    _write_snapshot(data_file)
    _add_zinger_search_pair(data_file)
    answerer = KfcQuestionAnswerer(
        data_file=data_file, index_file=tmp_path / "index.npz", semantic_enabled=False
    )

    results = answerer.search_items("ซิงเกอ เบอเกอร์", kind="menu", limit=2)

    assert len(results) == 2
    assert results[0][1]["name"] == "ซิงเกอร์ เบอร์เกอร์"
    assert results[1][1]["name"] == "ชุดซิงเกอร์เบอร์เกอร์"


def test_menu_search_accepts_similar_thai_final_consonant_typo(tmp_path):
    data_file = tmp_path / "menu.json"
    _write_snapshot(data_file)
    _add_bucket_search_items(data_file)
    answerer = KfcQuestionAnswerer(
        data_file=data_file, index_file=tmp_path / "index.npz", semantic_enabled=False
    )

    for typo in ("บักเกด", "บักเกจ"):
        results = answerer.search_items(typo, kind="menu", limit=10)

        assert [item["name"] for _, item in results[:3]] == [
            "สุขใจ บักเก็ต",
            "จุใจ บักเก็ต",
            "พอใจ บักเก็ต",
        ]
        assert all(score >= 0.94 for score, _ in results[:3])


def test_menu_search_finds_a_fuzzy_wingz_typo_and_prioritises_title_matches(tmp_path):
    data_file = tmp_path / "menu.json"
    _write_snapshot(data_file)
    _add_wingz_search_items(data_file)
    answerer = KfcQuestionAnswerer(
        data_file=data_file, index_file=tmp_path / "index.npz", semantic_enabled=False
    )

    results = answerer.search_items("วิงแสบ", kind="menu", limit=10)

    assert [item["name"] for _, item in results[:3]] == [
        "ไก่วิงซ์แซ่บ 3 ชิ้น",
        "วิงซ์แซ่บ 2 ชิ้น",
        "ชุดไก่วิงซ์แซ่บ 2 ชิ้น",
    ]
    assert all(score >= 0.94 for score, _ in results[:3])
    assert results[3][1]["name"] == "บักเก็ตสุดคุ้ม"
    assert results[3][0] < results[2][0]


def test_menu_search_rejects_a_name_with_no_close_match(tmp_path):
    data_file = tmp_path / "menu.json"
    _write_snapshot(data_file)
    answerer = KfcQuestionAnswerer(
        data_file=data_file, index_file=tmp_path / "index.npz", semantic_enabled=False
    )

    assert answerer.search_items("เมนูยานอวกาศที่ไม่มีอยู่", kind="menu", limit=2) == []
