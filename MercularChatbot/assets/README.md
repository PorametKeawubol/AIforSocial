# MercuMate visual assets

## Sources

- `source/mercumate-mascot.png`: user-provided mascot reference (`MercuMate.png`)
- `source/mercumate-cover.png`: user-provided brand/cover reference (`Cover.png`)
- `source/rich-menu-background-v1.png`: generated with Codex's built-in
  `image_gen` tool from both references; no text was requested in the generated
  bitmap so Thai labels could be rendered deterministically

The source files are kept unchanged. Final deliverables are versioned siblings:

- `rich_menu/mercumate-rich-menu-v1.svg`: previous deterministic menu source
- `rich_menu/mercumate-rich-menu-v1.jpg`: LINE-ready `2500×1686` image
- `rich_menu/mercumate-rich-menu-v2.svg`: production menu with promotions
- `rich_menu/mercumate-rich-menu-v2.jpg`: previous LINE-ready menu image
- `rich_menu/mercumate-rich-menu-v3.svg`: hierarchical category-menu source
- `rich_menu/mercumate-rich-menu-v3.jpg`: current LINE-ready `2500×1686` image
- `rich_menu/mercumate-imagemap-v1.jpg`: `1040×520` Imagemap master

Run `../scripts/render_rich_menu.sh` to rebuild the Rich Menu JPEG. It requires
ImageMagick and the Noto Sans Thai/Noto Sans font families.

## Image-generation prompt

```text
Use case: ui-mockup
Asset type: LINE rich menu background, large 3:2 landscape canvas
Primary request: create a premium futuristic six-zone menu background for
MercuMate, derived from the supplied mascot and cover art.
Input images: Image 1 is the mascot identity reference; Image 2 is the brand
palette, lighting, gaming/IT mood, and composition reference.
Scene/backdrop: deep navy-to-black high-tech showroom with electric cyan and
royal-blue light trails, subtle gaming gear silhouettes, elegant glass-panel
UI surfaces.
Subject: preserve the recognizable friendly white MercuMate robot with glossy
black face, cyan glowing eyes, wink, blue antenna orb, and white/blue shell.
Place the mascot as an integrated hero element without obscuring the six
actionable zones.
Style/medium: polished 3D commercial UI artwork, premium gaming and
consumer-tech aesthetic.
Composition/framing: exact 3:2 landscape intent; six clearly separated
interactive zones arranged as a clean 3-column by 2-row grid. Each zone must
retain a calm dark center area with strong contrast and generous empty space
for one icon and a short Thai label to be added later. Use consistent gutters
and safe margins. Keep all important art away from tile centers and boundaries.
Lighting/mood: energetic but friendly neon-blue rim light, crisp,
sophisticated, readable at mobile size.
Color palette: black, deep navy, electric cyan, royal blue, white highlights.
Constraints: no text, no letters, no numbers, no logos, no watermark; do not
invent extra characters; keep mascot identity consistent with Image 1; six
zones must be visually obvious and evenly aligned; strong mobile readability;
no tiny clutter.
Avoid: orange, red, purple dominance, illegible pseudo-text, distorted mascot,
crowded composition, flat generic dashboard.
```
