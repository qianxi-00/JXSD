# JavaScript 知识点总结

> 对应课案章节：**JavaScript（基础语法、DOM 操作、常用 DOM 操作、事件处理、在 HTML 中使用 JS 的两种方式）→ 综合示例**
>
> 配套文件（都在本目录下，直接双击用浏览器打开）：
>
> | 文件 | 覆盖的课案小节 | 特色 |
> |---|---|---|
> | `01_基础语法.html` | JavaScript 基础语法（变量 / 数据类型 / 条件 / 循环 / 函数） | 每个知识点都有**可点击运行的输出面板** |
> | `02_DOM操作.html` | DOM 操作 / 常用 DOM 操作 | 含**动态待办清单**与**数据生成表格**两个实战 |
> | `03_事件处理.html` | 事件处理 | 含**冒泡/捕获可视化**、**事件委托**、**表单校验** |
> | `04_外部JS/index.html` + `js/app.js` + `css/style.css` | 在 HTML 中使用 JS → 方式 2：外部 JS 文件（推荐） | 完整的外部文件工程结构 |
> | `05_综合示例/index.html` + `css/style.css` + `js/app.js` | 综合示例：三者结合 | **可用的小应用**：计数器 + 主题切换 + 待办清单 + 表单校验 |

> **说明：**课案里的示例统一使用 `alert()` 弹窗输出。为了不打断浏览体验（以及便于验证），
> 本目录的示例改成了把结果打印到页面上的**输出面板**并在注释中保留 `alert` 原始写法。
> 语法与思想完全一致，只是"输出到哪里"不同。调试时更推荐 `console.log()` + F12 Console。

---

## 1. JavaScript 是什么

**JavaScript（JS）是网页的编程语言，用于实现网页的交互功能。**

| 技术 | 负责 | 典型动作 |
|---|---|---|
| HTML | 网页结构 | 这里有个按钮 |
| CSS | 网页样式 | 按钮是蓝色的圆角 |
| **JavaScript** | **网页行为** | **点一下按钮，数字加一** |

> **一句话：**没有 JS，网页只是一份"会排版的文档"；有了 JS，网页才"活"。

---

## 2. 基础语法

### 变量

```javascript
let name = "张三";        // 可重新赋值
const PI = 3.14159;      // 不可重新赋值
var old = 1;             // ❌ 老写法，不要用（函数级作用域 + 变量提升）
```

| 关键字 | 能否重新赋值 | 作用域 | 建议 |
|---|---|---|---|
| `const` | ❌ | 块级 `{}` | **默认用它** |
| `let` | ✅ | 块级 `{}` | 需要改值时用 |
| `var` | ✅ | 函数级 | **不要用** |

> `const` 锁的是"变量指向"，不是"内容"：`const arr = [1]; arr.push(2);` 是合法的。

### 数据类型

```javascript
let str   = "Hello";              // 字符串 string
let num   = 25;                   // 数字 number（整数与小数同一种类型）
let bool  = true;                 // 布尔 boolean
let arr   = [1, 2, 3];            // 数组 array
let obj   = { name: "张三" };      // 对象 object
let nothing = null;               // 空值
let notDefined;                   // undefined 未定义
```

| 语法糖 | 说明 |
|---|---|
| `` `姓名: ${name}` `` | **模板字符串**（反引号 + `${}` 插值），比 `+` 拼接好用得多 |
| `typeof x` | 查看类型。⚠️ `typeof null === "object"`、`typeof [] === "object"` 是历史怪癖 |
| `Array.isArray(x)` | 判断是否为数组（`typeof` 判断不了） |

### 运算符

```javascript
7 / 3       // 2.333...  JS 除法永远返回小数
7 % 3       // 1         取余
2 ** 10     // 1024      幂
"5" + 3     // "53"  ⚠️ + 遇字符串变成拼接
"5" - 3     // 2     ⚠️ - 会把字符串转成数字
```

**比较运算符（重点）**

| 写法 | 含义 | 例子 |
|---|---|---|
| `==` | 只比值，**自动做类型转换** | `5 == "5"` → `true` |
| `===` | 比值**和**类型 | `5 === "5"` → `false` |

> **规矩：一律用 `===` / `!==`，永远不要用 `==` / `!=`。**
> `==` 的转换规则极其反直觉（`[] == false` 居然是 `true`），是 JS 最经典的 bug 来源。
> 另外 `NaN == NaN` 是 `false`（NaN 不等于自己），判断要用 `Number.isNaN()`。

**逻辑与短路**

```javascript
user && user.name        // user 为假时直接返回，不会报错（短路）
nickname || "匿名用户"     // 前面为假值时取默认值
0 || 100                 // 100  ⚠️ 0 被当成假值
0 ?? 100                 // 0    ✅ ?? 只在 null/undefined 时替换
user?.name               // undefined  ✅ 可选链，不报错
```

### 条件判断

```javascript
if (age >= 18) {
    console.log("成年人");
} else if (age >= 12) {
    console.log("青少年");
} else {
    console.log("未成年");
}

// 三元运算符：简写的 if-else
const label = age >= 18 ? "成年人" : "未成年";

// switch：适合"一个变量对多个确定值"
switch (day) {
    case 1: console.log("周一"); break;   // ⚠️ 忘了 break 会"贯穿"到下一个 case
    case 6:
    case 7: console.log("周末"); break;   // 多个 case 共用代码：故意不写 break
    default: console.log("其他");
}
```

### 循环

```javascript
for (let i = 0; i < 5; i++) { }          // 经典 for，需要下标时用
for (const item of arr) { }              // ★ 遍历数组的首选（拿到"值"）
for (const key in obj) { }               // 遍历对象（拿到"键名"）
for (const [i, v] of arr.entries()) { }  // 同时要下标和值
while (cond) { }                         // 次数不确定
do { } while (cond);                     // 至少执行一次

break;      // 跳出整个循环
continue;   // 跳过本轮，进入下一轮
```

**数组常用方法（比 for 循环更"声明式"，实战更常用）**

```javascript
const nums = [1, 2, 3, 4, 5];

nums.forEach(n => console.log(n));            // 逐个处理，无返回值
nums.map(n => n * 2);                         // [2,4,6,8,10] 一对一映射，长度不变
nums.filter(n => n % 2 === 0);                // [2,4]        筛选，长度可变
nums.reduce((acc, n) => acc + n, 0);          // 15           归约成一个值
nums.find(n => n > 3);                        // 4            找第一个，找不到返回 undefined
nums.some(n => n > 4);                        // true         有没有满足的
nums.every(n => n > 0);                       // true         是不是全部满足
[...nums].sort((a, b) => a - b);              // ⚠️ 数字排序必须传比较函数
```

> **两个关键点：**
> ① `map` / `filter` / `reduce` 都返回**新数组**，**不修改原数据**（更安全）；
> `push` / `pop` / `splice` / `sort` 会**修改原数组**。
> ② `[10, 9, 100, 2].sort()` 得到 `[10, 100, 2, 9]` —— 默认按**字符串**排序！
> 数字排序必须写 `.sort((a, b) => a - b)`。

### 函数

```javascript
// ① 函数声明（会提升，可以先调用后定义）
function add(a, b) { return a + b; }

// ② 函数表达式（不会提升）
const add2 = function (a, b) { return a + b; };

// ③ 箭头函数（最简洁，没有自己的 this）
const add3 = (a, b) => a + b;          // 单表达式可省略 {} 和 return
const square = x => x * x;             // 单个参数可省略括号
const hi = () => "Hi";                 // 无参数必须写 ()

// 默认参数
function greet(name = "陌生人") { return `你好，${name}`; }

// 剩余参数（收进一个真数组）
function sum(...nums) { return nums.reduce((a, b) => a + b, 0); }
```

> **闭包**：内层函数可以访问外层函数的变量，即使外层函数已经执行完毕。
> 用途：封装私有状态（例如计数器工厂，每次调用 +1 而不需要全局变量）。
>
> **函数没有 `return` 时返回 `undefined`。**

---

## 3. 在 HTML 中使用 JavaScript 的两种方式

### 方式 1：内联 `<script>` 标签

```html
<h1 id="title">点击次数：0</h1>
<button id="btn">点我</button>
<script>
    let count = 0;
    let btn = document.getElementById("btn");
    let title = document.getElementById("title");
    btn.onclick = function() {
        count++;
        title.textContent = "点击次数：" + count;
    };
</script>
```

### 方式 2：外部 `.js` 文件（**推荐**）

```html
<script src="js/app.js" defer></script>
```

| 对比项 | 内联 `<script>` | 外部 `.js` |
|---|---|---|
| 作用范围 | 单个页面 | **所有引用它的页面** |
| 浏览器缓存 | ❌ | ✅ |
| 可调试性 | 显示为 `index.html:行号` | ✅ 显示为 `app.js:行号`，能直接打断点 |
| 结论 | 极少量一次性代码 | **真实项目标准做法** |

### `<script>` 放在哪里？

| 位置 | 结果 |
|---|---|
| `<head>` 里（裸） | ❌ 此时 DOM 还没解析，`getElementById` 拿到 `null` → 报错 |
| `</body>` 前 | ✅ 传统稳妥做法 |
| `<head>` + `defer` | ✅ **现代推荐**：下载不阻塞，DOM 解析完按顺序执行 |

### `defer` / `async` 的区别

| 写法 | 下载 | 执行时机 |
|---|---|---|
| `<script>` | 阻塞 HTML 解析 | 下载完立即执行 |
| `<script defer>` | **不阻塞** | DOM 解析完，**按顺序**执行（推荐） |
| `<script async>` | **不阻塞** | 下载完**立即**执行（顺序不保证）→ 适合独立的第三方脚本 |

> ⚠️ `<script>` 不是空元素，必须写 `<script src="app.js"></script>`。

### 最常见的 JS 报错

```
Uncaught TypeError: Cannot read properties of null (reading 'onclick')
```

**原因**：`<script>` 在元素之前执行。**解决**：移到 `</body>` 前，或加 `defer`。

---

## 4. DOM 操作（重点）

**DOM（Document Object Model）= 浏览器把 HTML 解析成内存里的一棵对象树。**
JS 不能直接改 HTML 文本，只能改 DOM 对象。

> **核心口诀：获取元素 → 修改元素。**

### 获取元素

| 方法 | 说明 | 返回 |
|---|---|---|
| `document.getElementById("id")` | 通过 id 获取 | 元素 或 `null` |
| `document.querySelector(".class")` | 通过 CSS 选择器获取**第一个** | 元素 或 `null` |
| `document.querySelectorAll(".class")` | 通过选择器获取**所有** | `NodeList`（静态，有 `forEach`） |
| `document.getElementsByClassName("c")` | 按类名（老 API） | `HTMLCollection`（**动态**，无 `forEach`） |
| `document.getElementsByTagName("p")` | 按标签名（老 API） | `HTMLCollection` |
| `element.closest(".card")` | 从自己向上找最近的匹配祖先 | 元素 或 `null` |

> **推荐 `querySelector` / `querySelectorAll`** —— 它接受**任意合法 CSS 选择器**，
> 你在 CSS 章节学的知识可以直接复用：
> `"#id"`、`".class"`、`"div.card > p"`、`"input[type=text]"`、`"li:first-child"`。
>
> **注意**：`NodeList` 不是真数组（没有 `map` / `filter`），
> 要用 `[...list]` 或 `Array.from(list)` 转换。

### 修改内容

| 属性 | 说明 | 安全性 |
|---|---|---|
| `element.textContent` | 获取/设置**纯文本**，标签会原样显示 | ✅ 安全 |
| `element.innerHTML` | 获取/设置 **HTML**，字符串里的标签会被解析 | ⚠️ **有 XSS 风险** |
| `element.innerText` | 类似 textContent，但受 CSS 影响 | ✅ |
| `element.value` | 表单元素的值（**不是** textContent） | ✅ |

> **XSS 警告**：如果 `innerHTML` 的内容来自用户输入，攻击者可以注入
> `<img src=x onerror="...">` 并让它执行。
> **规矩：能用 `textContent` 就用 `textContent`。**

### 修改样式

```javascript
// 方式 1：行内样式（一次改一个属性，优先级最高、最难维护）
box.style.color = "red";
box.style.fontSize = "40px";           // ⚠️ CSS 的 font-size → JS 的 fontSize（小驼峰）
box.style.backgroundColor = "#eee";    // ⚠️ background-color → backgroundColor

// 方式 2：切换 class（★ 推荐：样式定义留在 CSS 里）
box.classList.add("active");
box.classList.remove("active");
box.classList.toggle("active");        // 有就删、没有就加（做开关最好用）
box.classList.contains("active");      // true / false
```

> **最常见错误**：把 `font-size` 直接写进 JS（`style.font-size`），
> 会被当成减法运算而报错或静默失效。

### 修改属性

```javascript
link.getAttribute("href");                        // 读
link.setAttribute("href", "https://python.org");  // 写
link.removeAttribute("target");                   // 删
link.hasAttribute("target");                      // 判断

// 有些属性可以直接点出来（等价但更常用）
input.value = "hello";
checkbox.checked = true;
img.src = "a.png";

// data-* 自定义属性用 dataset（自动中划线↔小驼峰转换）
// HTML: <div data-user-id="1001" data-role="admin">
div.dataset.userId;          // "1001"
div.dataset.role = "admin";  // 写回 HTML 的 data-role="admin"
```

> **区别**：`getAttribute("href")` 返回你写的那串字符；
> `link.href` 返回浏览器解析后的**绝对地址**。

### 增删元素

```javascript
// ① 创建
const li = document.createElement("li");
li.textContent = "新的内容";
li.className = "item";
li.dataset.id = "1001";

// ② 插入（★ 千万别忘这一步，否则元素只存在于内存里，页面上看不到）
list.appendChild(li);            // 插到最后
list.prepend(li);                // 插到最前
ref.before(li);                  // 插到 ref 前面
ref.after(li);                   // 插到 ref 后面
list.replaceChildren(newNode);   // 整体替换

// ③ 删除
li.remove();                     // 删自己
list.innerHTML = "";             // 清空容器

// ④ 克隆
const clone = li.cloneNode(true);   // true = 连子孙一起克隆
```

> **性能**：循环里反复 `appendChild` 会触发多次重排。
> 正确做法是用 `document.createDocumentFragment()` 先在内存里拼好，最后一次性插入。

### 在树里走动

```javascript
el.parentElement;               // 父元素
el.children;                    // 元素子节点（HTMLCollection）
el.childElementCount;           // 子元素个数
el.firstElementChild;           // 第一个子元素
el.nextElementSibling;          // 下一个兄弟元素
el.previousElementSibling;      // 上一个兄弟元素
el.closest(".card");            // 向上找最近的匹配祖先
```

> **区分**：`childNodes` 含文本/注释节点（缩进换行会产生一堆空白节点），
> `children` 只含元素节点。**遍历子元素请用 `children`。**

---

## 5. 事件处理（重点）

### 两种绑定方式

```javascript
// 方式一：onclick（DOM 0 级）
btn.onclick = function () { console.log("A"); };
btn.onclick = function () { console.log("B"); };   // ⚠️ 只有 B 生效，前一个被覆盖

// 方式二：addEventListener（DOM 2 级）★ 推荐
btn.addEventListener("click", function () { console.log("A"); });
btn.addEventListener("click", function () { console.log("B"); });   // ✅ A 和 B 都生效

// 移除：必须传同一个函数引用，所以不能用匿名函数
function handler() { }
btn.addEventListener("click", handler);
btn.removeEventListener("click", handler);      // ✅
// btn.removeEventListener("click", function(){});   // ❌ 无效
```

### 三个常用选项

```javascript
btn.addEventListener("click", fn, { once: true });      // 只触发一次，之后自动移除
window.addEventListener("scroll", fn, { passive: true }); // 承诺不 preventDefault，滚动更流畅
outer.addEventListener("click", fn, { capture: true });  // 在捕获阶段触发
```

### 事件对象 `event`

```javascript
element.addEventListener("click", function (event) {
    event.type;            // "click"
    event.target;          // ★ 真正被点中的元素（可能是子元素）
    event.currentTarget;   // ★ 绑定监听器的那个元素
    event.clientX/Y;       // 相对视口的坐标
    event.key;             // 键盘：按下的键（如 "Enter"、"a"、"ArrowUp"）
    event.code;            // 键盘：物理按键位置（如 "KeyA"，不受输入法影响）
    event.ctrlKey;         // 修饰键状态

    event.preventDefault();    // 阻止默认行为（跳转、提交表单、右键菜单）
    event.stopPropagation();   // 阻止继续冒泡
});
```

> **`target` = 谁被点了；`currentTarget` = 谁在听。** 两者不同说明事件冒泡上来了。

### 事件流：捕获 → 目标 → 冒泡

1. **捕获阶段**：从 `window` 向下传到目标元素的父级
2. **目标阶段**：事件到达被点击的元素本身
3. **冒泡阶段**：从目标元素向上传回 `window`

`addEventListener` **默认在冒泡阶段触发**，所以点内层元素时外层也会触发 ——
**这正是事件委托能成立的原因。**

### 事件委托（★ 最实用的技巧）

**问题**：列表有 100 项，每项一个删除按钮。逐个绑 = 100 个监听器；
而且**动态新增的项没有监听器，点了没反应**。

**解法**：监听器只绑在**父元素**上，靠 `event.target` 判断点了哪个子元素。

```javascript
// ✅ 只绑一次，绑在父容器上
document.querySelector(".list").addEventListener("click", function (event) {
    const delBtn = event.target.closest(".del");
    if (!delBtn) return;                 // 点的不是删除按钮，忽略
    const li = delBtn.closest("li");
    li.remove();
});
```

**优点**：① 监听器数量大幅减少；② **动态新增的元素自动生效**；③ 代码集中好维护。

### 常用事件速查

| 类别 | 事件 | 触发时机 |
|---|---|---|
| 鼠标 | `click` / `dblclick` | 单击 / 双击 |
| 鼠标 | `mouseenter` / `mouseleave` | 移入 / 移出（**不冒泡**，推荐） |
| 鼠标 | `mouseover` / `mouseout` | 移入 / 移出（**会冒泡**，在子元素间移动会反复触发） |
| 键盘 | `keydown` / `keyup` | 按下 / 松开 |
| 表单 | `input` | **每输入一个字符就触发**（实时校验 / 搜索） |
| 表单 | `change` | 值改变**且失去焦点**时触发 |
| 表单 | `submit` | 表单提交（**必须 `preventDefault`**） |
| 表单 | `focus` / `blur` | 获得 / 失去焦点 |
| 页面 | `DOMContentLoaded` | HTML 解析完成（<b>不等图片</b>，更快） |
| 页面 | `load` | 所有资源（含图片）都加载完 |
| 页面 | `scroll` / `resize` | 滚动 / 窗口尺寸变化 |

### `DOMContentLoaded` vs `window.onload`

| | `DOMContentLoaded` | `window.onload` |
|---|---|---|
| 时机 | HTML 解析完（不等图片） | 所有资源加载完 |
| 速度 | **更快** | 慢（图多要等很久） |
| 可否绑多个 | ✅ `addEventListener` 可绑多个 | ❌ 直接赋值会覆盖 |
| 建议 | **操作 DOM 用它** | 需要图片尺寸 / 做加载动画才用 |

---

## 6. 综合示例：三者结合

> **HTML 搭骨架 → CSS 化妆 → JS 加交互**

```html
<!-- HTML：结构 -->
<h1 id="title">点击次数：0</h1>
<button id="btn">点我</button>
<link rel="stylesheet" href="styles.css">
<script src="app.js"></script>
```

```css
/* CSS：样式 */
#title { color: #333; font-size: 24px; }
```

```javascript
// JavaScript：交互
let count = 0;
let btn = document.getElementById("btn");
let title = document.getElementById("title");
btn.onclick = function() {
    count++;
    title.textContent = "点击次数：" + count;
};
```

**三者的分工：**

- **HTML** 说「这里有一个 id 叫 `title` 的标题、一个 id 叫 `btn` 的按钮」—— 只描述结构
- **CSS** 说「`#title` 这个元素长成什么样」—— 只看选择器，不关心内容
- **JS** 说「找到 `btn`，当它被点击时，把 `title` 的文字改成新的计数」—— 靠 id 把两者连起来

完整可运行版本见 `05_综合示例/`，它把本目录所有知识点串成了一个**可用的小应用**：
计数器 + 亮暗主题切换（CSS 变量）+ 待办清单（数据驱动视图 + 事件委托 + localStorage）
+ 表单校验（`preventDefault` + 正则）+ 键盘快捷键。

---

## 7. 重点回顾 & 经典坑

### 重点

- **DOM 操作**：用 JS 动态修改网页内容。口诀「**获取元素 → 修改元素**」。
- **事件处理**：让网页响应用户操作。一律用 `addEventListener`。
- **事件委托**：一个监听器管一堆（含动态生成的）元素。
- **数据驱动视图**：只维护数据，变化后重新渲染 —— React / Vue 的核心思想。

### 经典坑

| 坑 | 现象 | 解决 |
|---|---|---|
| `<script>` 位置不对 | `Cannot read properties of null` | 移到 `</body>` 前，或加 `defer` |
| CSS 属性名没转小驼峰 | `style.font-size` 无效 | 写 `style.fontSize` |
| 表单没阻止默认行为 | 提交后页面刷新、状态全丢 | `event.preventDefault()` |
| `removeEventListener` 传匿名函数 | 移除不掉 | 用**具名函数**或保存函数引用 |
| 用 `var` 在循环里绑事件 | 所有回调看到同一个最终值 | 用 `let`，或用事件委托 |
| 循环里逐个 `appendChild` | 页面卡顿 | 用 `DocumentFragment` 一次性插入 |
| `innerHTML` 拼接用户输入 | XSS 漏洞 | 用 `textContent` |
| 用数组下标当 id | 删除中间项后错乱 | 用唯一 id（时间戳 / 自增序号） |
| 改了 JS 刷新没变化 | 浏览器用了缓存 | **Ctrl + F5** 强制刷新 |
| 箭头函数里的 `this` | 不是被点击的元素 | 需要 `this` 就用 `function`；或用 `event.currentTarget` |
