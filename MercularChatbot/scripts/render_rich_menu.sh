#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"
menu_dir="${project_dir}/assets/rich_menu"
source_svg="${menu_dir}/mercumate-rich-menu-v1.svg"
output_image="${menu_dir}/mercumate-rich-menu-v1.jpg"
render_tmp_dir="$(mktemp -d)"

cleanup() {
  rm -rf "${render_tmp_dir}"
}
trap cleanup EXIT

command -v convert >/dev/null
command -v identify >/dev/null

cd "${menu_dir}"
convert -background black "${source_svg}" "${render_tmp_dir}/base.png"
convert "${render_tmp_dir}/base.png" \
  -font 'Noto-Sans-Thai-Bold' \
  -pointsize 60 \
  -fill white \
  -gravity North \
  -annotate -782+715 'ค้นหาสินค้า' \
  -annotate +0+715 'เกมมิ่ง' \
  -annotate +782+715 'ออดิโอ' \
  -annotate -782+1275 'แก็ดเจ็ต' \
  -annotate +0+1275 'เดโมข้อความ' \
  -annotate +782+1275 'ช่วยเหลือ' \
  -strip \
  -interlace Plane \
  -quality 90 \
  -define jpeg:extent=950KB \
  "${output_image}"

dimensions="$(identify -format '%wx%h' "${output_image}")"
image_bytes="$(stat -c '%s' "${output_image}")"
if [[ "${dimensions}" != "2500x1686" ]]; then
  echo "Unexpected Rich Menu dimensions: ${dimensions}" >&2
  exit 1
fi
if (( image_bytes > 1000000 )); then
  echo "Rich Menu exceeds LINE's 1 MB limit: ${image_bytes} bytes" >&2
  exit 1
fi

echo "Rendered ${output_image} (${dimensions}, ${image_bytes} bytes)"
