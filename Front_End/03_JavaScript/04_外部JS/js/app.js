/* =============================================================================
 * 文件：js/app.js
 * 对应课案章节：JavaScript → 在HTML中使用JavaScript → 方式2：外部JS文件（推荐）
 *
 * 这个文件被同目录上一层的 index.html 通过下面这行引入：
 *     <script src="js/app.js" defer></script>
 *
 * 本文件包含三块内容：
 *   ① 课案原文的「点击计数器」示例（完整复现）
 *   ② 本项目额外补充的实战演示，用来把「基础语法 / DOM 操作 / 事件处理」
 *      三节的知识点串起来：
 *        - 待办清单：数据的增删改 + 视图同步 + 事件委托
 *        - 表单校验：input / submit 事件 + preventDefault + 正则
 *        - 实时搜索：input 事件 + filter + 键盘快捷键
 *
 * 【重要设计说明】所有代码都包在一个 init() 函数里，最后统一在
 * DOMContentLoaded 时调用一次。这样做的好处：
 *   1. 不会往 window 上挂一堆全局变量，避免和别的脚本冲突；
 *   2. 所有 DOM 操作都保证在"元素已经存在"之后执行；
 *   3. 只有一个入口，看代码时一眼就知道从哪开始读。
 * ========================================================================== */


/* =============================================================================
 * 一、课案原文示例：点击计数器
 * -----------------------------------------------------------------------------
 * 课案原文（在 </body> 前用内联 script）：
 *
 *     let count = 0;
 *     let btn = document.getElementById("btn");
 *     let title = document.getElementById("title");
 *
 *     btn.onclick = function() {
 *         count++;
 *         title.textContent = "点击次数：" + count;
 *     };
 *
 * 下面把它改写进函数里（同样的逻辑，更安全的作用域）。
 * ========================================================================== */
function initCounter() {
    // 用 let 声明可变的计数变量。它被下面的闭包"记住"，不会污染全局。
    let count = 0;

    const btn = document.getElementById("btn");
    const title = document.getElementById("title");
    const resetBtn = document.getElementById("btn-reset");

    // 防御性编程：如果页面里没有这个元素（比如别的页面也引入这个 js），
    // 就直接返回，避免 "Cannot read properties of null" 报错。
    if (!btn || !title) return;

    // 课案原文用的是 btn.onclick = ...
    // 这里改用 addEventListener，因为它可以绑多个监听器、也便于移除。
    btn.addEventListener("click", function () {
        count++;                                        // 自增
        title.textContent = "点击次数：" + count;         // 更新页面文字
    });

    // 额外加一个重置按钮（课案没有，方便反复演示）
    if (resetBtn) {
        resetBtn.addEventListener("click", function () {
            count = 0;
            title.textContent = "点击次数：0";
        });
    }
}


/* =============================================================================
 * 二、待办清单：数据驱动视图
 * -----------------------------------------------------------------------------
 * 核心思想（也是 React / Vue 的核心思想）：
 *
 *      数据（state）  ──渲染──▶  页面（DOM）
 *           ▲                       │
 *           └────── 用户操作 ────────┘
 *
 * 我们只维护一个 todos 数组，任何变化之后都调用一次 render()，
 * 由 render() 负责把数据"翻译"成 DOM。
 * 这样页面永远和数据一致，不会出现"删了数据但页面还在"这类不同步的 bug。
 * ========================================================================== */

/** 待办数据。每一项形如 { id: 1, text: "写作业", done: false } */
let todos = [
    { id: 1, text: "阅读课案「JavaScript 基础语法」", done: true },
    { id: 2, text: "动手改一改本页的 app.js", done: false }
];

/** 自增 id 生成器，保证每个待办有唯一 id（不要用数组下标当 id！） */
let todoSeq = 3;

function initTodoList() {
    const input = document.getElementById("todo-input");
    const addBtn = document.getElementById("todo-add");
    const clearDoneBtn = document.getElementById("todo-clear-done");
    const list = document.getElementById("todo-list");
    const stats = document.getElementById("todo-stats");

    // 页面上没有这些元素就直接跳过（本 js 只服务于 index.html）
    if (!input || !addBtn || !list) return;

    /**
     * 把 todos 数组渲染成 <ul> 里的 <li> 列表。
     * 每次数据变化后都要调用它，保证"数据 = 页面"。
     */
    function render() {
        // ① 先清空容器（最常用的一行）
        list.innerHTML = "";

        // ② 数据为空时给个友好提示
        if (todos.length === 0) {
            const empty = document.createElement("li");
            empty.className = "todo-empty";
            empty.textContent = "暂无待办，添加一条试试～";
            list.appendChild(empty);
        }

        // ③ 遍历数据，为每一条创建 DOM
        todos.forEach(function (todo) {
            const li = document.createElement("li");
            // 用 class 表达"已完成"状态（样式定义在 CSS 里）
            if (todo.done) li.classList.add("done");
            // 把 id 挂到 data-* 上，之后事件委托里靠它定位是哪一条
            li.dataset.id = todo.id;

            // 左侧：勾选框 + 文字
            const label = document.createElement("label");
            label.className = "todo-main";

            const checkbox = document.createElement("input");
            checkbox.type = "checkbox";
            checkbox.checked = todo.done;             // 受控：由数据决定勾选状态
            checkbox.dataset.action = "toggle";       // 供事件委托识别

            const span = document.createElement("span");
            span.textContent = todo.text;            // ⚠️ 用 textContent 防止 XSS

            label.appendChild(checkbox);
            label.appendChild(span);

            // 右侧：删除按钮
            const delBtn = document.createElement("button");
            delBtn.type = "button";
            delBtn.className = "mini-btn";
            delBtn.textContent = "删除";
            delBtn.dataset.action = "delete";         // 供事件委托识别

            li.appendChild(label);
            li.appendChild(delBtn);
            list.appendChild(li);
        });

        // ④ 更新统计信息
        const doneCount = todos.filter(function (t) { return t.done; }).length;
        stats.textContent = `共 ${todos.length} 条，已完成 ${doneCount} 条`;
    }

    /** 添加一条待办 */
    function addTodo() {
        // trim() 去掉首尾空白；用户只打了空格时不应被当成有效输入
        const text = input.value.trim();
        if (text === "") {
            input.focus();
            return;
        }
        todos.push({ id: todoSeq++, text: text, done: false });   // 改数据
        input.value = "";                                          // 清空输入框
        input.focus();
        render();                                                  // 同步视图
    }

    // 点按钮添加
    addBtn.addEventListener("click", addTodo);

    // 回车也能添加：这是表单类交互的标配体验
    input.addEventListener("keydown", function (event) {
        // event.key 是"按键的字符含义"，回车就是 "Enter"
        if (event.key === "Enter") {
            addTodo();
        }
    });

    // 清除所有已完成项
    clearDoneBtn.addEventListener("click", function () {
        // filter 返回【新数组】，只保留未完成的 —— 不修改原数组
        todos = todos.filter(function (t) { return !t.done; });
        render();
    });

    /* -----------------------------------------------------------------------
     * 事件委托：只在这一个 <ul> 上绑一个监听器，就能处理所有（含未来新增的）
     * 子元素的交互。这是本文件最重要的技巧。
     *
     * 为什么需要它？因为上面的 render() 每次都会把 <li> 全部重建，
     * 之前绑在旧 <li> 上的监听器会随元素一起被丢弃 —— 逐个绑是行不通的。
     * --------------------------------------------------------------------- */
    list.addEventListener("click", function (event) {
        // closest 从被点击的元素开始向上找最近的 [data-action]
        const actionEl = event.target.closest("[data-action]");
        if (!actionEl) return;                         // 点的不是按钮/勾选框
        if (actionEl.tagName === "INPUT" && actionEl.type === "checkbox") {
            // 勾选框是 click 事件里处理的（change 事件也能做，这里演示 click）
            handleToggle(actionEl);
            return;
        }
        const action = actionEl.dataset.action;

        // 找到这个元素所属的 <li>，再从 dataset 里读出 id
        const li = actionEl.closest("li");
        const id = Number(li.dataset.id);

        if (action === "delete") {
            // 从数据里删掉这一条，然后重新渲染
            todos = todos.filter(function (t) { return t.id !== id; });
            render();
        }
    });

    // 勾选框用 change 事件更语义化（键盘空格键也能触发）
    list.addEventListener("change", function (event) {
        const box = event.target.closest('input[type="checkbox"][data-action="toggle"]');
        if (!box) return;
        handleToggle(box);
    });

    /** 切换某一条待办的"已完成"状态 */
    function handleToggle(checkboxEl) {
        const li = checkboxEl.closest("li");
        const id = Number(li.dataset.id);
        // find 找到那一项并翻转它的 done
        const target = todos.find(function (t) { return t.id === id; });
        if (target) {
            target.done = checkboxEl.checked;
            render();
        }
    }

    // 首次进入页面先渲染一遍
    render();
}


/* =============================================================================
 * 三、表单校验：input / submit 事件 + preventDefault
 * ========================================================================== */

/** 简单的邮箱正则（够日常使用；严格校验应交给后端或更完整的正则） */
const EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

function initSignupForm() {
    const form = document.getElementById("signup-form");
    if (!form) return;

    const nameInput = document.getElementById("su-name");
    const emailInput = document.getElementById("su-email");
    const nameErr = document.getElementById("su-name-err");
    const emailErr = document.getElementById("su-email-err");
    const result = document.getElementById("form-result");

    /**
     * 设置一个字段的校验状态。
     * @param {HTMLInputElement} input   输入框
     * @param {HTMLElement} errEl        显示错误信息的元素
     * @param {string} message           错误信息；传空字符串表示通过
     */
    function mark(input, errEl, message) {
        if (message) {
            input.classList.remove("valid");
            input.classList.add("invalid");
            errEl.textContent = message;
        } else {
            input.classList.remove("invalid");
            input.classList.add("valid");
            errEl.textContent = "";
        }
    }

    /** 校验用户名，返回错误信息（通过则返回空字符串） */
    function validateName() {
        const value = nameInput.value.trim();
        if (value.length === 0) return "用户名不能为空";
        if (value.length < 2) return "用户名至少 2 个字符";
        return "";
    }

    /** 校验邮箱 */
    function validateEmail() {
        const value = emailInput.value.trim();
        if (value.length === 0) return "邮箱不能为空";
        // test() 用正则判断字符串是否匹配
        if (!EMAIL_PATTERN.test(value)) return "邮箱格式不正确";
        return "";
    }

    // input 事件：每输入一个字符就触发，用来做"实时"校验反馈
    nameInput.addEventListener("input", function () {
        // 只在用户已经开始输入之后才提示（没输入时不红）
        if (nameInput.value.length === 0) {
            mark(nameInput, nameErr, "");
            return;
        }
        mark(nameInput, nameErr, validateName());
    });

    emailInput.addEventListener("input", function () {
        if (emailInput.value.length === 0) {
            mark(emailInput, emailErr, "");
            return;
        }
        mark(emailInput, emailErr, validateEmail());
    });

    // submit 事件：点提交按钮 或 在输入框里按回车 时触发
    form.addEventListener("submit", function (event) {
        // ★★★ 这一行是本节的关键 ★★★
        // 表单的默认行为是"把数据提交到 action 指定的地址并刷新页面"。
        // 不阻止它，页面就会刷新，你写的所有 JS 状态全部丢失。
        event.preventDefault();

        const nameError = validateName();
        const emailError = validateEmail();
        mark(nameInput, nameErr, nameError);
        mark(emailInput, emailErr, emailError);

        // 有任何一个不通过就中断
        if (nameError || emailError) {
            result.className = "form-msg form-msg-error";
            result.textContent = "❌ 请修正上面的错误后再提交";
            return;
        }

        result.className = "form-msg form-msg-ok";
        result.textContent = `✅ 校验通过：${nameInput.value.trim()} / ${emailInput.value.trim()}（页面没有刷新）`;

        // 真实项目里这里会写：
        // fetch("/api/signup", {
        //     method: "POST",
        //     headers: { "Content-Type": "application/json" },
        //     body: JSON.stringify({ name: ..., email: ... })
        // });
    });

    // 重置按钮：清空表单和所有校验状态
    document.getElementById("su-reset").addEventListener("click", function () {
        form.reset();
        mark(nameInput, nameErr, "");
        mark(emailInput, emailErr, "");
        result.textContent = "";
        result.className = "";
    });
}


/* =============================================================================
 * 四、实时搜索 + 键盘快捷键
 * -----------------------------------------------------------------------------
 * 演示：input 事件 → filter 数据 → 重新渲染表格
 *      keydown 事件 → 全局快捷键（/ 聚焦搜索框，Esc 清空）
 * ========================================================================== */

/** 模拟从后端拿到的数据 */
const people = [
    { name: "张三", city: "北京", age: 25 },
    { name: "李四", city: "上海", age: 30 },
    { name: "王五", city: "广州", age: 28 },
    { name: "张伟", city: "深圳", age: 33 },
    { name: "赵敏", city: "北京", age: 27 },
    { name: "钱多多", city: "杭州", age: 22 }
];

function initSearch() {
    const input = document.getElementById("search-input");
    const table = document.getElementById("people-table");
    const hint = document.getElementById("search-hint");
    if (!input || !table) return;

    const tbody = table.querySelector("tbody");

    /**
     * 按关键字过滤并渲染表格。
     * @param {string} keyword 关键字（空字符串表示显示全部）
     */
    function render(keyword) {
        const kw = keyword.trim().toLowerCase();

        // filter 的经典用法：只要有一项为空就显示；否则做"包含"匹配
        const rows = people.filter(function (p) {
            if (kw === "") return true;
            return p.name.toLowerCase().includes(kw)
                || p.city.toLowerCase().includes(kw)
                || String(p.age).includes(kw);
        });

        // 用 DocumentFragment 批量构建，最后一次性插入 —— 性能更好
        // （在循环里反复 appendChild 会触发多次页面重排）
        const frag = document.createDocumentFragment();
        rows.forEach(function (p) {
            const tr = document.createElement("tr");
            [p.name, p.city, p.age].forEach(function (value) {
                const td = document.createElement("td");
                td.textContent = value;          // 依旧用 textContent，安全
                tr.appendChild(td);
            });
            frag.appendChild(tr);
        });

        // replaceChildren 会清空旧内容并插入新内容，比 innerHTML="" 再 append 更简洁
        tbody.replaceChildren(frag);

        // 没有结果时给出提示
        if (rows.length === 0) {
            hint.textContent = `没有匹配「${keyword}」的记录`;
        } else {
            hint.textContent = `共 ${rows.length} 条记录（总数据 ${people.length} 条）`;
        }
    }

    // input 事件：每敲一个字符就重新筛选
    input.addEventListener("input", function () {
        render(input.value);
    });

    // 键盘快捷键：/ 聚焦搜索框，Esc 清空
    // 绑在 document 上就是"全局快捷键"
    document.addEventListener("keydown", function (event) {
        // 如果焦点已经在输入框里，就不要抢按键（否则用户没法正常打字）
        const tag = document.activeElement ? document.activeElement.tagName : "";
        const isTyping = tag === "INPUT" || tag === "TEXTAREA";

        if (event.key === "/" && !isTyping) {
            event.preventDefault();      // 阻止浏览器把 "/" 当成"快速查找"
            input.focus();
            input.select();
        } else if (event.key === "Escape") {
            input.value = "";
            input.blur();                // 移开焦点
            render("");
        }
    });

    // 首次渲染
    render("");
}


/* =============================================================================
 * 五、统一的初始化入口
 * -----------------------------------------------------------------------------
 * 为什么还要用 DOMContentLoaded？
 *   因为 index.html 里的 <script> 已经加了 defer，理论上 DOM 已经就绪。
 *   但加上这层监听是"双重保险"：万一有人把这个 js 引到了别的页面、
 *   并且没写 defer，依然能正常工作。
 *
 * 注意：如果脚本是在 DOMContentLoaded 之后才执行的（例如 defer 脚本），
 *       document.readyState 会是 "interactive" 或 "complete"，
 *       此时再监听 DOMContentLoaded 是来不及的 —— 所以要判断一下。
 * ========================================================================== */
function init() {
    initCounter();
    initTodoList();
    initSignupForm();
    initSearch();
    console.log("[app.js] 所有模块初始化完成");
}

if (document.readyState === "loading") {
    // 还在解析 HTML，等 DOMContentLoaded
    document.addEventListener("DOMContentLoaded", init);
} else {
    // HTML 已解析完（defer 脚本就是这种情况），直接初始化
    init();
}
