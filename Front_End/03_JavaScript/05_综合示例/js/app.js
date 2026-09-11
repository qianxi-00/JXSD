/* =============================================================================
 * 文件：js/app.js
 * 对应课案章节：JavaScript → 综合示例：三者结合（JavaScript 部分）
 *
 * 这个文件是「HTML 搭骨架 → CSS 化妆 → JS 加交互」里的最后一块。
 * 被 ../index.html 通过 <script src="js/app.js" defer></script> 引入。
 *
 * 包含五个模块，每个模块一个 init 函数，最后统一在 initAll() 里调用：
 *   1. 课案原文的点击计数器
 *   2. 亮 / 暗主题切换（classList + data-* + localStorage）
 *   3. 待办清单（数据驱动视图 + 事件委托 + localStorage 持久化）
 *   4. 表单校验（input / submit 事件 + preventDefault + 正则）
 *   5. 键盘快捷键（/ 聚焦、Esc 清空、Ctrl+Enter 提交）
 *
 * 【为什么都包在函数里？】
 *   避免在全局作用域里到处声明变量（会和其他脚本撞名），
 *   同时保证所有 DOM 查询都发生在"元素已经存在"之后。
 * ========================================================================== */


/* =============================================================================
 * 小工具函数
 * ========================================================================== */

/**
 * 简写版 document.getElementById —— 少打很多字，代码更清爽。
 * @param {string} id 元素 id
 * @returns {HTMLElement|null}
 */
const $ = function (id) {
    return document.getElementById(id);
};

/**
 * 安全地读取 localStorage（有些浏览器隐私模式下会抛异常）。
 * @param {string} key   键
 * @param {*} fallback   读取失败或不存在时的默认值
 */
function loadJSON(key, fallback) {
    try {
        const raw = localStorage.getItem(key);
        // raw 为 null（从没存过）时返回默认值
        return raw === null ? fallback : JSON.parse(raw);
    } catch (err) {
        // JSON.parse 失败（数据被改坏了）或 localStorage 不可用
        console.warn("[app.js] 读取本地存储失败，使用默认值：", err);
        return fallback;
    }
}

/**
 * 安全地写入 localStorage。
 * @param {string} key   键
 * @param {*} value      值（会被 JSON 序列化）
 */
function saveJSON(key, value) {
    try {
        localStorage.setItem(key, JSON.stringify(value));
    } catch (err) {
        // 例如隐私模式、磁盘配额满
        console.warn("[app.js] 写入本地存储失败：", err);
    }
}


/* =============================================================================
 * 模块 1：课案的点击计数器
 * -----------------------------------------------------------------------------
 * 课案原文：
 *     let count = 0;
 *     let btn = document.getElementById("btn");
 *     let title = document.getElementById("title");
 *     btn.onclick = function() {
 *         count++;
 *         title.textContent = "点击次数：" + count;
 *     };
 *
 * 这里保持完全一样的逻辑，只是包进函数、换成 addEventListener。
 * ========================================================================== */
function initCounter() {
    const btn = $("btn");
    const title = $("title");
    const resetBtn = $("btn-reset");

    // 防御：如果这个页面没有这些元素，直接结束，不报错
    if (!btn || !title) return;

    let count = 0;   // 计数变量被下面的两个闭包共同"记住"

    // 课案用 btn.onclick，这里用 addEventListener（可以绑多个，更推荐）
    btn.addEventListener("click", function () {
        count += 1;
        title.textContent = "点击次数：" + count;
    });

    // 额外的重置按钮
    if (resetBtn) {
        resetBtn.addEventListener("click", function () {
            count = 0;
            title.textContent = "点击次数：0";
        });
    }
}


/* =============================================================================
 * 模块 2：亮 / 暗主题切换
 * -----------------------------------------------------------------------------
 * 实现思路：
 *   ① CSS 里用 :root 和 :root[data-theme="dark"] 定义了两套变量；
 *   ② JS 只要给 <html> 元素切换 data-theme 属性，整套配色立刻生效；
 *   ③ 选择结果写进 localStorage，下次打开还记得。
 *
 * 这个例子完美地说明了「CSS 变量」的价值：
 *   JS 只改了一个属性，几十个组件的颜色同时变化，一行样式代码都不用改。
 * ========================================================================== */
function initTheme() {
    const toggle = $("theme-toggle");
    const icon = $("theme-icon");
    const label = $("theme-label");
    if (!toggle) return;

    const STORAGE_KEY = "fe-demo-theme";

    /**
     * 应用一个主题。
     * @param {string} theme "light" 或 "dark"
     */
    function applyTheme(theme) {
        // dataset.theme = "dark" 会给 <html> 加上 data-theme="dark" 属性
        document.documentElement.dataset.theme = theme;

        // 同步按钮上的图标和文字
        if (icon)  icon.textContent = theme === "dark" ? "☀️" : "🌙";
        if (label) label.textContent = theme === "dark" ? "亮色" : "暗色";
    }

    // 初始化：优先用上次保存的，其次跟随操作系统偏好
    let saved = null;
    try {
        saved = localStorage.getItem(STORAGE_KEY);
    } catch (err) {
        console.warn("[app.js] 无法读取主题设置：", err);
    }

    if (saved !== "dark" && saved !== "light") {
        // window.matchMedia 可以查询系统的媒体特性，这里用来读"系统是否偏好暗色"
        const prefersDark = window.matchMedia
            && window.matchMedia("(prefers-color-scheme: dark)").matches;
        saved = prefersDark ? "dark" : "light";
    }
    applyTheme(saved);

    // 点击切换
    toggle.addEventListener("click", function () {
        const current = document.documentElement.dataset.theme === "dark" ? "dark" : "light";
        const next = current === "dark" ? "light" : "dark";
        applyTheme(next);
        try {
            localStorage.setItem(STORAGE_KEY, next);
        } catch (err) {
            console.warn("[app.js] 无法保存主题设置：", err);
        }
    });
}


/* =============================================================================
 * 模块 3：待办清单（数据驱动视图 + 事件委托 + 本地持久化）
 * ========================================================================== */
function initTodoList() {
    const input = $("todo-input");
    const addBtn = $("todo-add");
    const list = $("todo-list");
    const stats = $("todo-stats");
    const clearDoneBtn = $("todo-clear-done");
    const filterBox = $("todo-filters");

    if (!input || !addBtn || !list) return;

    const STORAGE_KEY = "fe-demo-todos";

    /* -------------------------------------------------------------------
     * 状态（state）：整个模块只维护这两份数据。
     * 每次数据变化后调用 render()，让页面和数据保持一致。
     * ----------------------------------------------------------------- */
    let todos = loadJSON(STORAGE_KEY, [
        { id: 1, text: "读完课案的 HTML / CSS / JavaScript 三章", done: true },
        { id: 2, text: "打开 F12，在 Elements 面板里看看本页的 DOM 树", done: false },
        { id: 3, text: "改一改这个 app.js，看看页面有什么变化", done: false }
    ]);
    let currentFilter = "all";     // all / active / done

    // 生成唯一 id：用当前时间戳递增，避免"数组下标当 id"带来的 bug
    let nextId = todos.reduce(function (max, t) {
        return Math.max(max, t.id || 0);
    }, 0) + 1;

    /** 把 todos 写进本地存储 */
    function persist() {
        saveJSON(STORAGE_KEY, todos);
    }

    /**
     * 根据 currentFilter 过滤出要显示的待办。
     * filter 返回【新数组】，不会改动原数组。
     */
    function visibleTodos() {
        if (currentFilter === "active") {
            return todos.filter(function (t) { return !t.done; });
        }
        if (currentFilter === "done") {
            return todos.filter(function (t) { return t.done; });
        }
        return todos;      // "all" 直接返回原数组
    }

    /* -------------------------------------------------------------------
     * render()：把数据"翻译"成 DOM。
     * 每次数据或筛选条件变了，都整体重渲染一遍 —— 逻辑简单，绝不会有
     * "数据删了页面还在"这类不同步的 bug。
     * ----------------------------------------------------------------- */
    function render() {
        // ① 清空列表容器
        list.innerHTML = "";

        const rows = visibleTodos();

        // ② 空状态
        if (rows.length === 0) {
            const empty = document.createElement("li");
            empty.className = "todo-empty";
            empty.textContent = currentFilter === "all"
                ? "还没有待办，在上面输入框里加一条吧～"
                : "这个筛选条件下没有内容";
            list.appendChild(empty);
        }

        // ③ 逐条创建 DOM
        rows.forEach(function (todo) {
            const li = document.createElement("li");
            li.dataset.id = todo.id;                    // 把 id 记在 data-* 上
            if (todo.done) li.classList.add("done");    // 完成状态用 class 表达

            // 左侧：勾选框 + 文字（包在 label 里，点文字也能勾选）
            const label = document.createElement("label");
            label.className = "todo-main";

            const box = document.createElement("input");
            box.type = "checkbox";
            box.checked = todo.done;
            box.dataset.action = "toggle";              // 供事件委托识别

            const span = document.createElement("span");
            span.className = "todo-text";
            span.textContent = todo.text;               // ⚠️ textContent，防止 XSS

            label.appendChild(box);
            label.appendChild(span);

            // 右侧：删除按钮
            const del = document.createElement("button");
            del.type = "button";
            del.className = "icon-btn";
            del.textContent = "✕";
            del.title = "删除这一条";
            del.dataset.action = "delete";              // 供事件委托识别

            li.appendChild(label);
            li.appendChild(del);
            list.appendChild(li);
        });

        // ④ 更新统计
        const doneCount = todos.filter(function (t) { return t.done; }).length;
        stats.textContent = `共 ${todos.length} 条 ｜ 未完成 ${todos.length - doneCount} 条 ｜ 已完成 ${doneCount} 条`;
    }

    /** 添加一条待办 */
    function addTodo() {
        const text = input.value.trim();     // 去掉首尾空白
        if (text === "") {
            input.focus();
            return;
        }
        todos.push({ id: nextId, text: text, done: false });
        nextId += 1;
        input.value = "";                    // 清空输入框
        input.focus();                       // 焦点留在输入框，方便连续输入
        persist();
        render();
    }

    addBtn.addEventListener("click", addTodo);

    // 回车快速添加（键盘事件）
    input.addEventListener("keydown", function (event) {
        if (event.key === "Enter") {
            addTodo();
        }
    });

    // 清除所有已完成
    clearDoneBtn.addEventListener("click", function () {
        todos = todos.filter(function (t) { return !t.done; });
        persist();
        render();
    });

    /* -------------------------------------------------------------------
     * 事件委托：只在这一个 <ul> 上绑监听器。
     * 因为 render() 每次都重建所有 <li>，逐个绑监听器是行不通的
     * （旧的监听器会随着元素被丢弃）。
     * ----------------------------------------------------------------- */
    list.addEventListener("click", function (event) {
        // closest 从被点击的元素向上找最近的 [data-action]
        const target = event.target.closest("[data-action]");
        if (!target) return;                 // 点的不是按钮/勾选框，忽略

        const li = target.closest("li");
        if (!li) return;
        const id = Number(li.dataset.id);    // dataset 取出来是字符串，要转成数字

        if (target.dataset.action === "delete") {
            todos = todos.filter(function (t) { return t.id !== id; });
            persist();
            render();
        }
    });

    // 勾选框用 change 事件（键盘空格也能触发，语义更准确）
    list.addEventListener("change", function (event) {
        const box = event.target.closest('input[type="checkbox"][data-action="toggle"]');
        if (!box) return;

        const id = Number(box.closest("li").dataset.id);
        const item = todos.find(function (t) { return t.id === id; });
        if (item) {
            item.done = box.checked;
            persist();
            render();
        }
    });

    // 筛选按钮组：也用事件委托（按钮数量以后可能变多）
    if (filterBox) {
        filterBox.addEventListener("click", function (event) {
            const btn = event.target.closest(".filter");
            if (!btn) return;

            currentFilter = btn.dataset.filter;

            // 用 classList 切换高亮：先全部去掉，再给当前这个加上
            filterBox.querySelectorAll(".filter").forEach(function (b) {
                b.classList.remove("active");
            });
            btn.classList.add("active");

            render();
        });
    }

    // 首屏渲染
    render();
}


/* =============================================================================
 * 模块 4：表单校验（input / submit 事件 + preventDefault）
 * ========================================================================== */

/** 邮箱正则：够日常使用。真正严格的校验应该交给后端。 */
const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

function initSignupForm() {
    const form = $("signup-form");
    if (!form) return;

    const nameInput = $("f-name");
    const emailInput = $("f-email");
    const citySelect = $("f-city");
    const hobbyBox = $("f-hobbies");
    const bioInput = $("f-bio");
    const bioCounter = $("f-bio-counter");
    const result = $("form-result");

    const nameErr = $("f-name-err");
    const emailErr = $("f-email-err");

    /**
     * 设置字段的校验状态类。
     * @param {HTMLInputElement} input 输入框
     * @param {HTMLElement} errEl      错误文字元素
     * @param {string} message         错误信息，空串表示通过
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

    /** 校验姓名，返回错误信息（通过返回空串） */
    function checkName() {
        const v = nameInput.value.trim();
        if (v.length === 0) return "姓名不能为空";
        if (v.length < 2)   return "姓名至少 2 个字符";
        return "";
    }

    /** 校验邮箱 */
    function checkEmail() {
        const v = emailInput.value.trim();
        if (v.length === 0) return "邮箱不能为空";
        if (!EMAIL_RE.test(v)) return "邮箱格式不正确";     // test() 用正则判断
        return "";
    }

    // input 事件：每敲一个字符就实时校验（用户不用等提交才知道错在哪）
    nameInput.addEventListener("input", function () {
        if (nameInput.value.length === 0) {
            mark(nameInput, nameErr, "");    // 还没开始输入，不标红
            return;
        }
        mark(nameInput, nameErr, checkName());
    });

    emailInput.addEventListener("input", function () {
        if (emailInput.value.length === 0) {
            mark(emailInput, emailErr, "");
            return;
        }
        mark(emailInput, emailErr, checkEmail());
    });

    // 简介字数实时统计
    if (bioInput && bioCounter) {
        const max = Number(bioInput.getAttribute("maxlength")) || 120;
        // 初始化显示（如果从缓存恢复了内容，这里也会算对）
        bioCounter.textContent = `还可输入 ${max - bioInput.value.length} 字`;
        bioInput.addEventListener("input", function () {
            const left = max - bioInput.value.length;
            bioCounter.textContent = `还可输入 ${left} 字`;
            // 快满的时候变个颜色提醒
            bioCounter.style.color = left <= 10 ? "var(--danger)" : "";
        });
    }

    /** 收集表单数据为一个对象 */
    function collectData() {
        // 勾选框组：用 querySelectorAll + filter 拿到被选中的那些
        const hobbies = [];
        if (hobbyBox) {
            hobbyBox.querySelectorAll('input[type="checkbox"]').forEach(function (box) {
                if (box.checked) hobbies.push(box.value);
            });
        }
        return {
            姓名: nameInput.value.trim(),
            邮箱: emailInput.value.trim(),
            城市: citySelect ? citySelect.value : "",
            兴趣: hobbies,
            简介: bioInput ? bioInput.value.trim() : ""
        };
    }

    /** 提交逻辑（校验 + 模拟提交） */
    function submit() {
        // 提交前把每个字段都校验一遍，这样用户一次就能看到所有错误
        const nameError = checkName();
        const emailError = checkEmail();
        mark(nameInput, nameErr, nameError);
        mark(emailInput, emailErr, emailError);

        if (nameError || emailError) {
            result.className = "form-result error";
            result.textContent = "❌ 请修正上面标红的字段后再提交";
            return;
        }

        const data = collectData();
        result.className = "form-result ok";
        // 用 JSON.stringify 美化展示 + innerHTML 做换行
        result.innerHTML = "✅ 校验通过，表单数据如下（页面没有刷新）：<br><code>"
            + JSON.stringify(data, null, 2).replace(/\n/g, "<br>")
            + "</code>";

        // 真实项目里这里改成：
        // fetch("/api/signup", { method: "POST", body: JSON.stringify(data) });
    }

    // ★★★ submit 事件：必须 preventDefault，否则页面会刷新 ★★★
    form.addEventListener("submit", function (event) {
        event.preventDefault();       // 阻止浏览器"提交并刷新页面"的默认行为
        submit();
    });

    // 重置
    const resetBtn = $("f-reset");
    if (resetBtn) {
        resetBtn.addEventListener("click", function () {
            form.reset();             // form.reset() 是浏览器内置的清空表单方法
            mark(nameInput, nameErr, "");
            mark(emailInput, emailErr, "");
            result.className = "form-result";
            result.textContent = "";
            if (bioCounter && bioInput) {
                bioCounter.textContent = `还可输入 ${Number(bioInput.getAttribute("maxlength")) || 120} 字`;
                bioCounter.style.color = "";
            }
        });
    }

    // 键盘快捷键：在简介框里按 Ctrl + Enter 直接提交
    if (bioInput) {
        bioInput.addEventListener("keydown", function (event) {
            // event.ctrlKey 判断有没有按住 Ctrl；Mac 上习惯用 Meta（⌘），所以两个都判断
            if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
                event.preventDefault();     // 防止在 textarea 里插入换行
                submit();
            }
        });
    }
}


/* =============================================================================
 * 模块 5：全局键盘快捷键
 * -----------------------------------------------------------------------------
 *   /        聚焦到待办输入框（很多网站用 / 打开搜索）
 *   Escape   清空并让输入框失焦
 * ========================================================================== */
function initShortcuts() {
    const todoInput = $("todo-input");
    if (!todoInput) return;

    document.addEventListener("keydown", function (event) {
        // 判断当前焦点是不是在输入控件里。在输入框里打字时不能抢按键，
        // 否则用户想输入 "/" 却触发了快捷键。
        const active = document.activeElement;
        const isTyping = active && (
            active.tagName === "INPUT"
            || active.tagName === "TEXTAREA"
            || active.tagName === "SELECT"
        );

        if (event.key === "/" && !isTyping) {
            event.preventDefault();       // 阻止浏览器把 "/" 当成"快速查找"
            todoInput.focus();
            todoInput.select();
        } else if (event.key === "Escape") {
            todoInput.value = "";
            todoInput.blur();
        }
    });
}


/* =============================================================================
 * 统一入口
 * -----------------------------------------------------------------------------
 * 注意：index.html 里的 <script> 已经加了 defer，所以执行到这里时 DOM 一定
 *       已经解析完成。但为了"万一有人没写 defer"，这里再判断一次 readyState。
 * ========================================================================== */
function initAll() {
    initCounter();
    initTheme();
    initTodoList();
    initSignupForm();
    initShortcuts();
    console.log("[app.js] 综合示例的 5 个模块全部初始化完成");
}

if (document.readyState === "loading") {
    // 还在解析 HTML，等 DOMContentLoaded
    document.addEventListener("DOMContentLoaded", initAll);
} else {
    // HTML 已解析完（defer 脚本的情况），直接初始化
    initAll();
}
