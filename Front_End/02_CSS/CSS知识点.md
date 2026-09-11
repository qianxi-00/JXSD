# CSS 知识点总结

> 对应课案章节：**CSS（语法、常用属性、盒子模型、Flexbox 布局、在 HTML 中使用 CSS）**
>
> 配套文件（都在本目录下，直接双击用浏览器打开）：
>
> | 文件 | 覆盖的课案小节 | 特色 |
> |---|---|---|
> | `01_CSS语法.html` | CSS 语法 / 在 HTML 中使用 CSS 的三种方式 / 选择器 / 优先级 | 每条选择器都有真实效果 |
> | `02_常用属性.html` | CSS 常用属性：文字样式 / 背景与边框 / 尺寸 | 属性速查表 + 现场演示 |
> | `03_盒子模型.html` | 盒子模型 | **带可拖动滑块的盒子模型可视化器**，实时算出总尺寸 |
> | `04_Flex布局.html` | Flexbox 布局 | **可点击按钮切换** justify-content / align-items / flex-wrap / flex-direction |
> | `05_外部样式/index.html` + `05_外部样式/css/style.css` | 在 HTML 中使用 CSS → 方式 2：外部 CSS 文件（推荐） | 完整可复用的真实项目级样式表（含 CSS 变量、响应式） |

---

## 1. CSS 是什么

**CSS（Cascading Style Sheets，层叠样式表）用于控制网页的外观和布局。**

> **HTML 管内容结构，CSS 管样式表现。**

| 技术 | 负责 | 类比 |
|---|---|---|
| HTML | 内容结构 —— 这里是什么 | 砖墙、房间划分 |
| **CSS** | **样式表现 —— 它长什么样** | 刷墙、贴壁纸、摆家具 |
| JavaScript | 行为交互 —— 它能干什么 | 水电、开关、电梯 |

**为什么要分开？** 一个页面可能有 100 个同样的"卡片"，样式写一次就全部生效；
改主题色只改一个地方。写死在 HTML 里就要改 100 处。

---

## 2. CSS 语法

```css
h1 {
    color: red;
    font-size: 24px;
}
 ↑        ↑    ↑
 选择器    属性  值
    └──── 声明 ────┘（以分号结尾）
 └──────── 一条规则 ────────┘（用花括号包起来）
```

| 组成部分 | 本例 | 作用 |
|---|---|---|
| **选择器** selector | `h1` | 给「谁」上样式 |
| **属性** property | `color`、`font-size` | 改「什么」 |
| **值** value | `red`、`24px` | 改成「多少」 |
| **声明** declaration | `color: red;` | 「属性: 值;」构成一条声明 |

### 新手三个语法错误

1. 忘了分号：`color: red font-size: 24px;`（后面那条整条被丢弃）
2. 用了等号：`color = red;`（CSS 用冒号）
3. 花括号/分号位置错乱

> **CSS 出错时不报错、不提示，只是那条不生效。** F12 → Styles 面板是唯一可靠的排查工具。

---

## 3. 在 HTML 中使用 CSS 的三种方式

### 方式 1：行内样式

```html
<p style="color: red; font-size: 20px;">文字</p>
```

- ✅ 最直接，优先级最高
- ❌ 无法复用、无法统一修改、结构样式混杂
- 用于：临时调试、邮件 HTML

### 方式 2：`<style>` 标签（内部样式表）

```html
<head>
    <style>
        h1 { color: blue; }
        .highlight { background: yellow; }
    </style>
</head>
```

- ✅ 页面内可复用
- ❌ 只作用于当前 HTML 文件
- 用于：单文件 Demo、本页独有样式

### 方式 3：外部 CSS 文件（**推荐**）

```html
<link rel="stylesheet" href="css/style.css">
```

| `<link>` 属性 | 作用 |
|---|---|
| `rel="stylesheet"` | 声明「这是一个样式表」，**不能省** |
| `href` | CSS 文件路径（相对路径最佳） |
| `media` | 只在特定媒体下应用，如 `media="print"`、`media="(max-width: 600px)"` |

- ✅ 一处修改，处处生效；浏览器会**缓存**；HTML 干净
- ✅ 真实项目的标准做法

| 对比项 | 行内 style | `<style>` | 外部 .css |
|---|---|---|---|
| 作用范围 | 单个元素 | 单个页面 | **所有引用它的页面** |
| 浏览器缓存 | ❌ | ❌ | ✅ |
| 适合场景 | 临时调试 | 单文件 Demo | **生产项目** |

### `<link>` vs `@import`

| | `<link>` | `@import` |
|---|---|---|
| 加载 | 与 HTML 解析**并行** | 必须等外层 CSS 解析完 → **串行，更慢** |
| 结论 | **优先使用** | 少数内聚场景才用 |

### 路径写法

| 写法 | 名称 | 结果 |
|---|---|---|
| `css/style.css` | 相对路径（相对当前 HTML 文件） | ✅ **推荐** |
| `../css/style.css` | 上一级目录 | ✅ 子页面引用公共样式常用 |
| `/css/style.css` | 站点根目录 | ⚠️ `file://` 直接打开时**失效** |
| `F:\x\style.css` | 本机绝对路径 | ❌ 换电脑立刻失效 |

---

## 4. 选择器与优先级

### 常用选择器

| 写法 | 名称 | 含义 | 权重 |
|---|---|---|---|
| `*` | 通配 | 所有元素 | 0 |
| `h1` | 标签 | 所有该标签 | 1 |
| `.card` | 类 | 所有 `class="card"` | 10 |
| `#main` | id | `id="main"` 的那一个 | 100 |
| `[type="text"]` | 属性 | 按属性值筛选 | 10 |
| `.a, .b` | 并集 | 共用一套规则（逗号分隔） | — |
| `.a .b` | 后代 | A 里**所有** B（可跨层） | 相加 |
| `.a > .b` | 子代 | A 的**直接子元素** B（只隔一层） | 相加 |
| `.a + .b` | 相邻兄弟 | 紧跟在 A 后面的那一个 B | 相加 |
| `a:hover` | 伪类 | 鼠标悬停状态 | 10 |
| `p::before` | 伪元素 | 在元素内部生成内容 | 1 |

### 优先级（谁说了算）

1. **重要性**：带 `!important` 的胜出（尽量别用）
2. **选择器权重**：行内 style(1000) > id(100) > 类/伪类/属性(10) > 标签/伪元素(1) > 通配(0)
   - 多个选择器权重**相加**：`.card h2` = 10 + 1 = 11
3. **书写顺序**：权重相同时，**后写的赢**（这就是"层叠"）

> **调试口诀：** 样式不生效 → ① 看 Styles 面板里这条规则有没有被划掉；② 看是否被行内 style 覆盖；③ 检查选择器有没有写错。

---

## 5. CSS 常用属性

### 文字样式

| 属性 | 作用 | 常用值 |
|---|---|---|
| `color` | 文字颜色 | 关键字 / `#hex` / `rgb()` / `rgba()` / `hsl()` |
| `font-size` | 字号 | `16px` / `1.2rem` / `120%` |
| `font-family` | 字体族（可写多个，逗号分隔，从左往右找） | `"Microsoft YaHei", Arial, sans-serif` |
| `font-weight` | 字重 | `400`=normal、`700`=bold |
| `text-align` | 水平对齐（作用于块级元素内的行内内容） | `left`/`center`/`right`/`justify` |
| `line-height` | 行高 | 无单位数字（字号倍数，如 `1.75`）**最推荐** |
| `text-decoration` | 装饰线 | `none`/`underline`/`line-through` |
| `letter-spacing` | 字间距 | `2px` |
| `text-shadow` | 文字阴影 | `2px 2px 4px rgba(0,0,0,.4)` |
| `text-transform` | 大小写 | `uppercase`/`lowercase`/`capitalize` |

> **常见误区：** `text-align: center` 让**元素内部的行内内容**居中；
> 想让**盒子本身**水平居中，要用 `margin: 0 auto;`。

### 尺寸单位

| 单位 | 含义 | 特点 |
|---|---|---|
| `px` | 像素 | 绝对单位，最常用 |
| `%` | 百分比 | 相对**父容器**的同名尺寸 |
| `vw` / `vh` | 视口宽/高的 1% | `100vw` = 整个窗口宽度 |
| `rem` | 根元素字号的倍数 | 适合响应式，改一个根字号整站缩放 |
| `em` | 当前元素字号的倍数 | 会层层相乘，容易算错 |
| `auto` | 自动 | 块级元素宽度默认撑满父容器 |

- `max-width` / `min-height`：宽度自适应但**不超过上限**
- `width: 100%` 是"不超过父容器"，`max-width: 100%` 通常更符合预期

### 背景

| 属性 | 作用 | 常用值 |
|---|---|---|
| `background-color` | 背景色 | 任意颜色 / `transparent` |
| `background-image` | 背景图 | `url("a.png")` 或 `linear-gradient(...)` |
| `background-size` | 背景图尺寸 | `cover`（铺满裁切）/ `contain`（完整留白）/ `100% 100%` |
| `background-repeat` | 是否平铺 | `repeat` / `no-repeat` |
| `background-position` | 位置 | `center` / `top left` |
| `background` | 以上全部简写 | `#eee url("a.png") no-repeat center / cover` |

> **推荐用渐变代替纯装饰性背景图**：浏览器实时计算，零请求、零流量、任意分辨率都清晰。

### 边框

```css
border: 1px solid black;             /* 宽度 样式 颜色 */
border-bottom: 1px solid #e1e4e8;    /* 只要下边框（做分割线） */
border-radius: 8px;                  /* 圆角，50% = 正圆 */
box-shadow: 0 4px 12px rgba(0,0,0,.25);   /* 水平 垂直 模糊 颜色 */
box-shadow: inset 0 0 8px rgba(0,0,0,.2); /* inset = 内阴影 */
```

`border-style` 可选：`solid`（实线）/ `dashed`（虚线）/ `dotted`（点线）/ `double`（双线）/ `none`（无）。

### 颜色写法

| 写法 | 示例 | 说明 |
|---|---|---|
| 关键字 | `red` | 只有 140 多个预设名 |
| 6 位 hex | `#3498db` | `#RRGGBB`，最常用 |
| 3 位 hex | `#f39` = `#ff3399` | 每两位相同才能简写 |
| `rgb()` | `rgb(231,76,60)` | 十进制 0~255 |
| `rgba()` | `rgba(231,76,60,.35)` | 第 4 个参数是透明度 0~1 |
| `hsl()` | `hsl(145,63%,42%)` | 色相/饱和度/亮度，调同色系深浅最方便 |

> **取色技巧：** F12 → Styles 面板里点任意颜色小方块，会弹出取色器，
> 可视化调整并实时复制出 hex / rgb / hsl 各种写法。

---

## 6. 盒子模型（重点）

**每个 HTML 元素都是一个矩形盒子**，由内到外四层：

```
┌─────────────────── margin（外边距，永远透明）───────────────────┐
│  ┌──────────────── border（边框）────────────────┐              │
│  │  ┌──────────── padding（内边距，吃背景色）───┐ │              │
│  │  │  ┌─────── content（内容区）────────┐     │ │              │
│  │  │  │  width × height               │     │ │              │
│  │  │  └───────────────────────────────┘     │ │              │
│  │  └────────────────────────────────────────┘ │              │
│  └─────────────────────────────────────────────┘              │
└───────────────────────────────────────────────────────────────┘
```

| 层次 | 名称 | 作用 | 特点 |
|---|---|---|---|
| 1（最内） | **content** | 放内容，宽高由 `width`/`height` 决定 | — |
| 2 | **padding** | 内容与边框之间的空白，在边框**以内** | **会**被背景色覆盖；**不能为负**；**可以点击** |
| 3 | **border** | 边框线 | — |
| 4（最外） | **margin** | 盒子与**其他盒子**之间的距离 | **不会**被背景色覆盖；**可以为负**；点不到 |

### 四边简写

```css
padding: 10px;                    /* 四边相同 */
padding: 10px 20px;               /* 上下 / 左右 */
padding: 10px 20px 30px;          /* 上 / 左右 / 下 */
padding: 10px 20px 30px 40px;     /* 上 右 下 左（顺时针） */
margin-left: auto;                /* 左外边距自动 → 右对齐技巧 */
```

> **口诀：** 「上右下左，顺时针」；两个值是「先竖后横」。

### 尺寸计算公式（**最大陷阱**）

```css
.demo-box {
    width: 160px;
    padding: 16px;
    border: 6px solid red;
    margin: 24px;
    box-sizing: content-box;   /* 默认值 */
}
```

| 层 | 左右各 | 累计宽度 |
|---|---|---|
| content | — | 160 px |
| padding | 16 px | 160 + 32 = 192 px |
| border | 6 px | 192 + 12 = **204 px** ← 盒子自身宽度 |
| margin | 24 px | 204 + 48 = **252 px** ← 占用的总横向空间 |

> **「我设了 width: 200px，为什么它占了 300px？」** 答案就在这里。

### 推荐做法：`box-sizing: border-box`

```css
*, *::before, *::after {
    box-sizing: border-box;
}
```

- `content-box`（默认）：**width 只管内容**，加 padding/border 越撑越大
- `border-box`（推荐）：**width 管到底**，加 padding/border 会向内挤压内容，总宽不变

> 做「两个卡片各占 50% 并排」这类布局时，不用 `border-box` 几乎必然溢出换行。
> **这是现代 CSS 的第一条必备规则。**

### padding 还是 margin？

| 问题 | 答案 |
|---|---|
| 让**内容**离边框远点 | `padding` |
| 让**这个盒子**离别的盒子远点 | `margin` |

### margin 的经典陷阱

1. **外边距合并（margin collapse）**：相邻元素的**上下** `margin` 会**合并取较大值**，不是相加。
   - 相邻兄弟之间会合并
   - 父元素与第一个/最后一个子元素之间也会合并（"父元素高度塌陷"）
   - **左右 margin 不会合并**
   - 破解：给父元素加 `padding: 1px` / `overflow: hidden` / `display: flex`
2. **行内元素的垂直 margin 无效**：`<span>` 的 `margin-top/bottom` 不生效，
   `padding-top/bottom` 会画出背景但不会撑开行高。
   要给元素设垂直间距，先确保它是 `block` 或 `inline-block`。

---

## 7. Flexbox 弹性布局（重点）

### 核心原理

**父容器设置 `display: flex;` 后，它的直接子元素自动变成"弹性项目"，按规则排列。**

```html
<div class="container">
    <div class="box">1</div>
    <div class="box">2</div>
    <div class="box">3</div>
</div>
```

```css
.container {
    display: flex;
    justify-content: center;   /* 主轴（默认水平）居中 */
    align-items: center;       /* 交叉轴（默认垂直）居中 */
    gap: 10px;                 /* 子元素之间间隔 10px */
    height: 100px;
    background-color: #eee;
}
.box {
    width: 60px;
    height: 60px;
    background-color: #3498db;
    color: white;
    display: flex;             /* 方块自己也是容器，用来居中数字 */
    justify-content: center;
    align-items: center;
}
```

### 两根轴（理解 Flex 的钥匙）

| 概念 | 默认方向 | 谁控制方向 | 谁负责对齐 |
|---|---|---|---|
| **主轴** main axis | 水平 → | `flex-direction` | `justify-content` |
| **交叉轴** cross axis | 垂直 ↓ | 由主轴决定（与主轴垂直） | `align-items` / `align-content` |

> **⚠️ 不要死记"水平/垂直"，要记"主轴/交叉轴"。**
> `flex-direction: column` 时，两根轴互换，`justify-content` 变成管垂直，
> `align-items` 变成管水平。

### 容器属性

**`justify-content`（主轴对齐）**

| 值 | 效果 |
|---|---|
| `flex-start` | 所有元素靠起点（默认） |
| `center` | 所有元素居中 |
| `flex-end` | 所有元素靠终点 |
| `space-between` | 两端贴边，中间均分（**导航栏左 logo 右菜单就用它**） |
| `space-around` | 每项左右各留一半间距（两端是中间的一半） |
| `space-evenly` | 所有间距完全相等（含两端） |

**`align-items`（交叉轴对齐）**

| 值 | 效果 |
|---|---|
| `stretch` | **拉伸填满交叉轴（默认值！）** |
| `flex-start` | 靠起点（默认水平时=靠顶） |
| `center` | 居中 |
| `flex-end` | 靠终点（默认水平时=靠底） |
| `baseline` | 按文字基线对齐 |

> **`stretch` 是默认值**，这是「子元素高度不受控 / 百分比高度不生效」的最常见原因。
> 想让子元素自己决定高度，先设 `align-items: flex-start;`。

**`flex-direction`**

| 值 | 主轴 | `justify-content` 管 | `align-items` 管 |
|---|---|---|---|
| `row`（默认） | 水平，左→右 | 水平 | 垂直 |
| `row-reverse` | 水平，右→左 | 水平（反向） | 垂直 |
| `column` | 垂直，上→下 | **垂直** | **水平** |
| `column-reverse` | 垂直，下→上 | **垂直**（反向） | **水平** |

**`flex-wrap`**

| 值 | 效果 |
|---|---|
| `nowrap`（默认） | 不换行，子元素被**压缩** |
| `wrap` | 放不下就换行 |
| `wrap-reverse` | 换行，行序反向 |

配合 `align-content` 控制**多行之间**的分布（仅换行且有剩余高度时有效）。

**`gap`**：子元素之间的间距。
比给每个子元素写 `margin` 干净得多（不会在两端多出间距）。

### 项目属性（写在子元素上）

| 属性 | 作用 |
|---|---|
| `flex-grow` | 有剩余空间时按比例分配。0=不分（默认），1=分 |
| `flex-shrink` | 空间不够时是否压缩。1=可缩（默认），0=**不许缩** |
| `flex-basis` | 分配空间前的基准尺寸，如 `200px` 或 `auto` |
| `flex` | 简写：`flex: grow shrink basis` |
| `align-self` | 单独覆盖父容器的 `align-items` |
| `order` | 改变排列顺序（数字越小越靠前，默认 0），**不改 HTML 结构** |

```css
.item-a { flex: 1; }     /* = 1 1 0%   ：等分剩余空间（做等宽列） */
.item-b { flex: auto; }  /* = 1 1 auto ：按内容大小分配并伸展 */
.item-c { flex: none; }  /* = 0 0 auto ：固定大小，不伸不缩（做侧边栏） */

/* 三栏布局：左右固定，中间自适应 */
.layout { display: flex; gap: 12px; }
.side   { flex: 0 0 150px; }
.main   { flex: 1; }

/* 响应式卡片墙（黄金公式，不需要媒体查询） */
.cards  { display: flex; flex-wrap: wrap; gap: 16px; }
.card   { flex: 1 1 220px; }
```

### Flex 常见坑

1. **`flex` 只对直接子元素生效**，孙子元素不受影响。
2. **flex 容器的子元素，其 `float` / `vertical-align` 会失效。**
3. **`margin: auto` 在 flex 里有奇效**：`margin-left: auto;` 可把某项单独推到最右。
4. **`position: absolute` 的子元素不参与 flex 布局。**
5. **`align-items: stretch` 是默认值**，会导致子元素被拉伸到容器高度。

---

## 8. 重点回顾

- CSS 负责**样式表现**，与 HTML（结构）、JS（行为）分离。
- 一条规则 = `选择器 { 属性: 值; }`；**CSS 出错不报错**，靠 F12 的 Styles 面板排查。
- 引入方式：行内 style > `<style>` 标签 > 外部 `.css`（**推荐**）。
- 优先级：`!important` > 行内 > id > 类 > 标签；相同则**后写的赢**。
- **盒子模型**：content → padding → border → margin。
  **盒子自身宽 = width + padding×2 + border×2**（默认 content-box）；
  用 `box-sizing: border-box` 让 width 管到底 —— **现代 CSS 第一条必备规则**。
- 相邻**上下** margin 会**合并取较大值**；行内元素的垂直 margin 无效。
- **Flexbox**：父容器 `display: flex`，子元素自动排列。
  - 记**主轴/交叉轴**，不要记水平/垂直
  - `justify-content` 管主轴，`align-items` 管交叉轴（默认 `stretch`！）
  - `flex: 1` 等分空间，`flex: 0 0 150px` 固定侧边栏，`gap` 管间距
- 真实项目用 **CSS 变量**集中管理配色/间距，用 **@media** 做响应式适配。
