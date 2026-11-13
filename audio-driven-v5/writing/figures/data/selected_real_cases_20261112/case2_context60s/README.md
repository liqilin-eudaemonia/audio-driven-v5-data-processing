# S0557 附近一分钟素材

源区间 **206.525–266.525 秒**，对应原视频03:26.525–04:26.525。原选pair在本分钟的 **26.765–33.235秒**。

- `both_tracks_highlighted.svg/pdf/png`：双方真实波形，同一幅度尺度；蓝色标出已选6.47秒片段。
- `both_tracks_plain.svg/pdf/png`：无蓝色高亮；另有每条音轨的透明背景SVG/PDF/PNG，方便自行剪图。
- `speaker_60s.wav`、`listener_60s.wav`：48kHz浮点无损独立音轨（沿用整轨归一化），**门控静音前**。一分钟内角色可能变化，未把原pair的listener静音规则延伸到整分钟。
- `speaker_original_view_60s.mp4`、`listener_original_view_60s.mp4`：双方分别的原始完整画面，2160×3840、原帧率，未套用6.47秒的固定人脸框。H264 CRF18重编码，保留各自独立声音。
- `both_views_60s_preview.mp4`：左右并排预览。画面左A/P2517、右B/P2254；声音左声道A、右声道B。个体原尺寸视频用于正式剪辑，并排版用于浏览。
- `selected_pair_times.csv`：源视频与这一分钟内的位置对应；`provenance.json`保存来源及媒体参数。

A是原选pair的speaker、B是原选pair的listener；一分钟内不要把这两个角色视为始终不变。原来6.47秒的净化视频、最终listener裁框、caption和Event素材仍在上一级case2，未作更改。
