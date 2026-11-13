# 透明背景音轨图

所有PNG均为真正RGBA透明背景，左上角alpha=0；SVG保留可编辑矢量。不是把背景涂成白色。ALPHA_AUDIT.json记录每张图的透明通道检查。

- case1：S1371的speaker音轨。
- case2：S0557的speaker、listener净化前与净化后独立波形，以及带坐标的对照图。
- case2_context60s：附近一分钟双方独立波形和对齐图。
- case2_role_swap/role_swapped_176370_185297：P2254主讲、P2517倾听的波形。
- case2_role_swap/later_reply_269053_273876：后续双方讲话重叠的波形。

独立长条图不含坐标或文字，方便放进论文流程图；两行对照图保留文字坐标，但画布背景透明。时间、形状、颜色和幅度尺度均不改，只移除白色画布。
