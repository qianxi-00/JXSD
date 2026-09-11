/* =============================================================================
 * 文件：06_综合实战/js/app.js
 * 对应课案章节：JavaScript（基础语法 / DOM 操作 / 事件处理 / 在 HTML 中使用 JS）
 *              + 重点回顾：「JavaScript 实现网页交互」
 *
 * 本文件是「个人主页小站」的全部交互逻辑，被 ../index.html 通过
 *     <script src="js/app.js" defer></script>
 * 引入。
 *
 * 包含七个模块，每个模块一个 init 函数，最后统一在 initAll() 里按顺序调用：
 *   1. initTheme()        亮 / 暗主题切换（classList + data-* + localStorage）
 *   2. initMobileMenu()   移动端汉堡菜单（classList.toggle）
 *   3. initNavHighlight() 滚动时高亮当前区块对应的导航项（scroll 事件）
 *   4. initBackToTop()    回到顶部按钮的显示/隐藏与点击
 *   5. initWorks()        作品列表：数据驱动视图 + 事件委托 + 分类筛选
 *   6. initContactForm()  联系表单：实时校验 + 提交前总校验 + preventDefault
 *   7. initMisc()         零碎功能：知识点计数动画、Ctrl+Enter 快捷提交
 *
 * 【为什么所有代码都包在函数里？】
 *   ① 不往 window 上挂一堆全局变量，避免和其它脚本冲突；
 *   ② 所有 DOM 查询都保证发生在"元素已经存在"之后；
 *   ③ 只有一个入口 initAll()，看代码时知道从哪开始读。
 * ========================================================================== */


/* =============================================================================
 * 通用小工具
 * ========================================================================== */

/**
 * 简写版 getElementById —— 少打很多字，代码更清爽。
 * @param {string} id 元素 id
 * @returns {HTMLElement|null}
 */
function $(id) {
    return document.getElementById(id);
}

/**
 * 安全地读 localStorage。
 * 用 try/catch 包起来是因为：浏览器的隐私模式 / 禁用 Cookie 时，
 * 访问 localStorage 会直接抛异常，不能让整个脚本挂掉。
 * @param {string} key
 * @param {string|null} fallback 读取失败时的默认值
 * @returns {string|null}
 */
function safeGetItem(key, fallback = null) {
    try {
        const value = localStorage.getItem(key);
        return value === null ? fallback : value;
    } catch (err) {
        console.warn('[app.js] 读取 localStorage 失败：', err);
        return fallback;
    }
}

/**
 * 安全地写 localStorage。
 * @param {string} key
 * @param {string} value
 */
function safeSetItem(key, value) {
    try {
        localStorage.setItem(key, value);
    } catch (err) {
        console.warn('[app.js] 写入 localStorage 失败：', err);
    }
}


/* =============================================================================
 * 模块 1：亮 / 暗主题切换
 * -----------------------------------------------------------------------------
 * 实现思路：
 *   ① CSS 里用 :root 和 :root[data-theme="dark"] 定义了两套变量；
 *   ② JS 只需要给 <html> 元素切换 data-theme 属性，整套配色立刻生效；
 *   ③ 选择结果写进 localStorage，下次打开还记得。
 *
 * 这个例子最能说明 CSS 变量的价值：JS 只改了一个属性，
 * 几十处颜色同时变化，一行 CSS 都不用改。
 * ========================================================================== */
function initTheme() {
    const toggle = $('theme-toggle');
    const icon = $('theme-icon');
    if (!toggle || !icon) return;

    const STORAGE_KEY = 'fe-site-theme';

    /**
     * 应用一个主题。
     * @param {'light'|'dark'} theme
     */
    function applyTheme(theme) {
        // dataset.theme = 'dark' 会给 <html> 加上 data-theme="dark" 属性
        document.documentElement.dataset.theme = theme;
        icon.textContent = theme === 'dark' ? '☀️' : '🌙';
    }

    // 初始化：优先用上次保存的，其次跟随操作系统的偏好
    let saved = safeGetItem(STORAGE_KEY);
    if (saved !== 'dark' && saved !== 'light') {
        // window.matchMedia 可以查询系统的媒体特性
        const prefersDark = window.matchMedia
            && window.matchMedia('(prefers-color-scheme: dark)').matches;
        saved = prefersDark ? 'dark' : 'light';
    }
    applyTheme(saved);

    // 点击切换
    toggle.addEventListener('click', function () {
        const current = document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light';
        const next = current === 'dark' ? 'light' : 'dark';
        applyTheme(next);
        safeSetItem(STORAGE_KEY, next);
    });
}


/* =============================================================================
 * 模块 2：移动端汉堡菜单
 * ========================================================================== */
function initMobileMenu() {
    const toggle = $('menu-toggle');
    const nav = $('main-nav');
    if (!toggle || !nav) return;

    /** 展开 / 收起菜单，并同步按钮的无障碍属性 */
    function toggleMenu() {
        const isOpen = nav.classList.toggle('is-open');
        toggle.classList.toggle('is-open', isOpen);
        // aria-expanded 让屏幕阅读器知道菜单当前是展开还是收起
        toggle.setAttribute('aria-expanded', String(isOpen));
    }

    toggle.addEventListener('click', toggleMenu);

    // 点击任意导航链接后自动收起菜单（手机上体验更好）
    nav.addEventListener('click', function (event) {
        if (event.target.closest('.nav-link')) {
            nav.classList.remove('is-open');
            toggle.classList.remove('is-open');
            toggle.setAttribute('aria-expanded', 'false');
        }
    });

    // 按 Esc 也能关闭菜单（键盘可访问性）
    document.addEventListener('keydown', function (event) {
        if (event.key === 'Escape' && nav.classList.contains('is-open')) {
            nav.classList.remove('is-open');
            toggle.classList.remove('is-open');
            toggle.setAttribute('aria-expanded', 'false');
        }
    });
}


/* =============================================================================
 * 模块 3：滚动时高亮当前区块对应的导航项
 * -----------------------------------------------------------------------------
 * 做法：监听 scroll 事件，遍历每个区块，找出"当前视口顶部附近"的那一个，
 *       给它对应的 .nav-link 加上 .is-active 类。
 * ========================================================================== */
function initNavHighlight() {
    const navLinks = document.querySelectorAll('.nav-link[data-nav]');
    if (navLinks.length === 0) return;

    const header = $('site-header');

    // 把导航链接整理成 [{ id, link }] 的形式，方便后面查找
    const items = [];
    navLinks.forEach(function (link) {
        const section = $(link.dataset.nav);
        if (section) {
            items.push({ id: link.dataset.nav, link: link, section: section });
        }
    });
    if (items.length === 0) return;

    /**
     * 根据滚动位置更新高亮。
     * 用 requestAnimationFrame 节流：scroll 事件触发极其频繁，
     * 每个像素都算一遍会浪费性能。这里保证每帧最多算一次。
     */
    let ticking = false;

    function update() {
        ticking = false;

        // 页头高度：用于计算"内容真正露出来的位置"
        const headerHeight = header ? header.offsetHeight : 0;
        // 视口高度的 1/3 处作为判定线：区块顶部越过这条线就算"当前区块"
        const judgeLine = window.scrollY + headerHeight + window.innerHeight / 3;

        let currentId = items[0].id;
        items.forEach(function (item) {
            // offsetTop 是元素相对文档顶部的位置
            if (item.section.offsetTop <= judgeLine) {
                currentId = item.id;
            }
        });

        // 滚动到底部时，强制高亮最后一个（否则最后一块可能永远高亮不到）
        if (window.innerHeight + window.scrollY >= document.body.offsetHeight - 4) {
            currentId = items[items.length - 1].id;
        }

        items.forEach(function (item) {
            item.link.classList.toggle('is-active', item.id === currentId);
        });

        // 页头滚动后加阴影
        if (header) {
            header.classList.toggle('is-scrolled', window.scrollY > 8);
        }
    }

    window.addEventListener('scroll', function () {
        if (!ticking) {
            // requestAnimationFrame 让更新发生在下一帧，避免同一帧里算很多次
            window.requestAnimationFrame(update);
            ticking = true;
        }
    }, { passive: true });   // passive: true 表示"我不会 preventDefault"，滚动更流畅

    // 窗口尺寸变化时也要重算
    window.addEventListener('resize', update, { passive: true });

    // 页面刚打开时先算一次
    update();
}


/* =============================================================================
 * 模块 4：回到顶部按钮
 * ========================================================================== */
function initBackToTop() {
    const button = $('back-to-top');
    if (!button) return;

    // 滚动超过 400px 才显示按钮
    window.addEventListener('scroll', function () {
        button.classList.toggle('is-visible', window.scrollY > 400);
    }, { passive: true });

    button.addEventListener('click', function () {
        // behavior: 'smooth' 让滚动有动画效果
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });
}


/* =============================================================================
 * 模块 5：作品列表（数据驱动视图 + 事件委托 + 分类筛选）
 * -----------------------------------------------------------------------------
 * 这是本文件最重要的模块，它演示了现代前端框架的核心思想：
 *
 *      数据（works 数组）  ──renderWorks()──▶  页面（DOM）
 *              ▲                                  │
 *              └──────── 用户点击筛选 ──────────────┘
 *
 * 注意 index.html 里的 #work-grid 是【空的】——
 * 所有卡片都是这里用 createElement 动态生成的。
 * ========================================================================== */
function initWorks() {
    const grid = $('work-grid');
    const emptyTip = $('work-empty');
    const filterBar = $('work-filters');
    if (!grid) return;

    /* -------------------------------------------------------------------
     * 数据：每个作品一条记录。
     * 在真实项目里，这份数据通常来自后端接口（fetch / axios）。
     * ----------------------------------------------------------------- */
    const works = [
        {
            id: 1,
            title: '数据看板原型',
            desc: '用 Streamlit + pandas 做的销售数据看板，包含指标卡、趋势图、明细表和 CSV 导出。',
            category: 'data',
            categoryLabel: '数据应用',
            cover: '📊',
            from: '#4f46e5',
            to: '#818cf8',
            tags: ['Streamlit', 'pandas', 'Altair'],
        },
        {
            id: 2,
            title: '图书管理系统',
            desc: '带登录与角色权限的图书借阅系统：图书增删改查、借还书、借阅记录、JSON 持久化。',
            category: 'tool',
            categoryLabel: '小工具',
            cover: '📚',
            from: '#0891b2',
            to: '#22d3ee',
            tags: ['Streamlit', 'JSON', '权限'],
        },
        {
            id: 3,
            title: '响应式个人主页',
            desc: '就是你现在看的这个页面。纯手写 HTML / CSS / JS，无框架、无外部依赖、可离线运行。',
            category: 'web',
            categoryLabel: '网页',
            cover: '🌐',
            from: '#ec4899',
            to: '#f9a8d4',
            tags: ['HTML', 'CSS', 'Flexbox'],
        },
        {
            id: 4,
            title: '待办清单小程序',
            desc: '用原生 JS 实现的数据驱动清单：增删改查、筛选、localStorage 持久化、键盘快捷键。',
            category: 'web',
            categoryLabel: '网页',
            cover: '✅',
            from: '#16a34a',
            to: '#4ade80',
            tags: ['JavaScript', 'DOM', 'localStorage'],
        },
        {
            id: 5,
            title: 'CSV 数据清洗工具',
            desc: '上传 CSV，自动识别缺失值、重复行、异常值，给出清洗建议并导出一份干净的数据。',
            category: 'data',
            categoryLabel: '数据应用',
            cover: '🧹',
            from: '#7c3aed',
            to: '#a78bfa',
            tags: ['pandas', 'numpy', 'file_uploader'],
        },
        {
            id: 6,
            title: '颜色主题生成器',
            desc: '输入一个主色，自动生成一套协调的配色方案，并输出成可直接粘贴的 CSS 变量。',
            category: 'tool',
            categoryLabel: '小工具',
            cover: '🎨',
            from: '#ea580c',
            to: '#fb923c',
            tags: ['color_picker', 'CSS 变量'],
        },
    ];

    /** 当前筛选条件 */
    let currentFilter = 'all';

    /**
     * 创建一个作品卡片 DOM 节点。
     * 这是 DOM 操作的典型流程：createElement → 设置属性/内容 → 组装 → 返回。
     * @param {object} work 作品数据
     * @returns {HTMLElement}
     */
    function createWorkCard(work) {
        // ① 卡片外层
        const card = document.createElement('article');
        card.className = 'work-card';
        // 把 id 和 category 挂到 data-* 上，方便事件委托里读取
        card.dataset.id = work.id;
        card.dataset.category = work.category;

        // ② 顶部渐变封面（用 CSS 变量把两个颜色传进去）
        const cover = document.createElement('div');
        cover.className = 'work-cover';
        cover.style.setProperty('--cover-from', work.from);
        cover.style.setProperty('--cover-to', work.to);
        cover.textContent = work.cover;

        // ③ 正文区
        const body = document.createElement('div');
        body.className = 'work-body';

        const title = document.createElement('h3');
        title.className = 'work-title';
        // ★ 用 textContent 而不是 innerHTML —— 内容可控时更安全，也不会有解析开销
        title.textContent = work.title;

        const desc = document.createElement('p');
        desc.className = 'work-desc';
        desc.textContent = work.desc;

        // ④ 标签区
        const tags = document.createElement('div');
        tags.className = 'work-tags';
        work.tags.forEach(function (tagText) {
            const tag = document.createElement('span');
            tag.className = 'work-tag';
            tag.textContent = tagText;
            tags.appendChild(tag);
        });

        // ⑤ 组装：把子节点塞进父节点
        body.appendChild(title);
        body.appendChild(desc);
        body.appendChild(tags);

        card.appendChild(cover);
        card.appendChild(body);

        return card;
    }

    /**
     * 根据当前筛选条件，重新渲染整个作品网格。
     * 每次数据或筛选条件变化都整体重渲染 —— 逻辑简单，绝不会出现"数据变了页面没变"。
     */
    function renderWorks() {
        // ① 先清空容器
        grid.innerHTML = '';

        // ② 按条件过滤（filter 返回新数组，不改原数组）
        const list = currentFilter === 'all'
            ? works
            : works.filter(function (work) { return work.category === currentFilter; });

        // ③ 用 DocumentFragment 批量插入 —— 性能更好
        //    （在循环里反复 appendChild 会触发多次页面重排）
        const fragment = document.createDocumentFragment();
        list.forEach(function (work) {
            fragment.appendChild(createWorkCard(work));
        });
        grid.appendChild(fragment);

        // ④ 空状态提示
        if (emptyTip) {
            emptyTip.hidden = list.length > 0;
        }
    }

    /* -------------------------------------------------------------------
     * 事件委托：只在这一个父容器上绑一个监听器，处理所有筛选按钮
     * （包括将来可能动态新增的按钮）。
     * ----------------------------------------------------------------- */
    if (filterBar) {
        filterBar.addEventListener('click', function (event) {
            // closest 从被点的元素向上找最近的 .filter-btn
            const button = event.target.closest('.filter-btn');
            if (!button) return;                 // 点的不是按钮

            currentFilter = button.dataset.filter;

            // 用 classList 切换高亮：先全部去掉，再给当前这个加上
            filterBar.querySelectorAll('.filter-btn').forEach(function (btn) {
                btn.classList.remove('is-active');
            });
            button.classList.add('is-active');

            renderWorks();
        });
    }

    // 点作品卡片时，在卡片上叠加一行"提示"，演示事件委托读取 data-*
    grid.addEventListener('click', function (event) {
        const card = event.target.closest('.work-card');
        if (!card) return;
        const id = Number(card.dataset.id);
        const work = works.find(function (item) { return item.id === id; });
        if (work) {
            console.log('[作品卡片被点击]', work);
            // 用 toast 风格的临时提示（纯 DOM + 定时器实现，不依赖任何库）
            showToast(`你点击了「${work.title}」（详情已打印到 F12 控制台）`);
        }
    });

    // 首次渲染
    renderWorks();
}

/**
 * 显示一个 2.4 秒后自动消失的浮层提示。
 * 用 createElement 现场创建节点、插入 body、定时移除 ——
 * 这是"动态生成 DOM"最简单的实战例子。
 * @param {string} text 提示文字
 */
function showToast(text) {
    const toast = document.createElement('div');
    toast.textContent = text;
    // 直接写行内样式（这里为了不改动 CSS 文件；真实项目建议加一个 CSS 类）
    Object.assign(toast.style, {
        position: 'fixed',
        left: '50%',
        bottom: '76px',
        transform: 'translateX(-50%) translateY(10px)',
        padding: '10px 18px',
        borderRadius: '10px',
        background: 'var(--primary)',
        color: '#fff',
        fontSize: '14px',
        boxShadow: '0 10px 30px rgba(0,0,0,.25)',
        opacity: '0',
        transition: 'opacity .25s ease, transform .25s ease',
        zIndex: '200',
        pointerEvents: 'none',
        maxWidth: '90vw',
        textAlign: 'center',
    });

    document.body.appendChild(toast);

    // 下一帧再改透明度，让 transition 有"从无到有"的动画
    // requestAnimationFrame 保证浏览器已经完成一次布局
    window.requestAnimationFrame(function () {
        toast.style.opacity = '1';
        toast.style.transform = 'translateX(-50%) translateY(0)';
    });

    // 2.4 秒后淡出并移除节点
    window.setTimeout(function () {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(-50%) translateY(10px)';
        // 等过渡结束再真正从 DOM 里删掉
        window.setTimeout(function () {
            toast.remove();      // element.remove() 删除自身
        }, 300);
    }, 2400);
}


/* =============================================================================
 * 模块 6：联系表单校验
 * -----------------------------------------------------------------------------
 * 完整流程：
 *   输入时（input 事件）   → 实时校验，立刻给出反馈
 *   提交时（submit 事件）  → 再整体校验一遍（防止用户没触发 input 就提交）
 *                          → 用 preventDefault() 阻止页面刷新
 *                          → 通过则把数据渲染到右侧结果面板
 * ========================================================================== */
function initContactForm() {
    const form = $('contact-form');
    if (!form) return;

    const nameInput = $('cf-name');
    const emailInput = $('cf-email');
    const topicSelect = $('cf-topic');
    const messageInput = $('cf-message');
    const messageCount = $('cf-message-count');
    const resultBox = $('form-result');
    const resetButton = $('cf-reset');

    const nameError = $('cf-name-error');
    const emailError = $('cf-email-error');
    const messageError = $('cf-message-error');

    /* --- 校验规则 ---------------------------------------------------------
       把规则抽成独立的小函数，返回"错误信息字符串"（通过时返回空串）。
       这样输入时和提交时可以复用同一套规则，绝不会出现"输入时通过、提交时说错"。
       ------------------------------------------------------------------- */

    /** 一个简单够用的邮箱正则 */
    const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

    function checkName() {
        const value = nameInput.value.trim();
        if (value.length === 0) return '姓名不能为空';
        if (value.length < 2) return '姓名至少 2 个字符';
        if (value.length > 20) return '姓名不能超过 20 个字符';
        return '';
    }

    function checkEmail() {
        const value = emailInput.value.trim();
        if (value.length === 0) return '邮箱不能为空';
        if (!EMAIL_RE.test(value)) return '邮箱格式不正确，应形如 name@example.com';
        return '';
    }

    function checkMessage() {
        const value = messageInput.value.trim();
        if (value.length === 0) return '留言不能为空';
        if (value.length < 10) return `留言至少 10 个字（当前 ${value.length} 个字）`;
        return '';
    }

    /**
     * 给一个字段设置校验状态。
     * @param {HTMLInputElement|HTMLTextAreaElement} input
     * @param {HTMLElement} errorEl
     * @param {string} message 错误信息（空串 = 通过）
     * @param {boolean} showValid 是否给通过的状态加绿框（输入过程中不加，避免满屏绿）
     */
    function markField(input, errorEl, message, showValid) {
        if (message) {
            input.classList.remove('is-valid');
            input.classList.add('is-invalid');
            // 设置 aria-invalid，屏幕阅读器能读到"这个字段有错误"
            input.setAttribute('aria-invalid', 'true');
        } else {
            input.classList.remove('is-invalid');
            input.classList.toggle('is-valid', Boolean(showValid) && input.value.trim() !== '');
            input.removeAttribute('aria-invalid');
        }
        if (errorEl) {
            errorEl.textContent = message;
        }
    }

    /* --- 实时校验：input 事件每敲一个字符就触发 ------------------------- */
    nameInput.addEventListener('input', function () {
        // 还没开始输入时不报错（否则一进页面就满屏红字）
        markField(nameInput, nameError, nameInput.value.length === 0 ? '' : checkName(), false);
    });

    emailInput.addEventListener('input', function () {
        markField(emailInput, emailError, emailInput.value.length === 0 ? '' : checkEmail(), false);
    });

    messageInput.addEventListener('input', function () {
        const length = messageInput.value.length;
        messageCount.textContent = `${length} / 200`;
        // 接近上限时把计数变红提醒
        messageCount.style.color = length >= 190 ? 'var(--danger)' : '';
        markField(messageInput, messageError,
            messageInput.value.length === 0 ? '' : checkMessage(), false);
    });

    // change 事件：值改变且失去焦点时触发（和 input 的区别：要失焦才触发）
    emailInput.addEventListener('change', function () {
        console.log('[change 事件] 邮箱输入完成：', emailInput.value);
    });

    // blur 事件：失去焦点时做一次完整校验（这时用户"填完了"，可以放心标绿）
    nameInput.addEventListener('blur', function () {
        markField(nameInput, nameError, checkName(), true);
    });
    emailInput.addEventListener('blur', function () {
        markField(emailInput, emailError, checkEmail(), true);
    });

    /* --- 提交 ------------------------------------------------------------- */
    /**
     * 执行一次完整的校验 + 提交。
     * 抽成函数是因为"点提交按钮"和"按 Ctrl+Enter"都要用它。
     */
    function submitForm() {
        const nameMsg = checkName();
        const emailMsg = checkEmail();
        const messageMsg = checkMessage();

        // 提交时强制显示所有字段的状态（包括绿框）
        markField(nameInput, nameError, nameMsg, true);
        markField(emailInput, emailError, emailMsg, true);
        markField(messageInput, messageError, messageMsg, true);

        const errors = [nameMsg, emailMsg, messageMsg].filter(Boolean);

        if (errors.length > 0) {
            // 一次性把所有问题列出来，用户改一遍就能全改对
            resultBox.className = 'form-result is-error';
            resultBox.innerHTML = '';
            const p = document.createElement('p');
            p.textContent = `❌ 有 ${errors.length} 个问题需要修正（页面没有刷新）：`;
            resultBox.appendChild(p);

            const ul = document.createElement('ul');
            errors.forEach(function (msg) {
                const li = document.createElement('li');
                li.textContent = msg;
                ul.appendChild(li);
            });
            resultBox.appendChild(ul);
            return;
        }

        // ---- 通过：把数据渲染到结果面板 ----
        const data = {
            姓名: nameInput.value.trim(),
            邮箱: emailInput.value.trim(),
            主题: topicSelect.value,
            留言: messageInput.value.trim(),
        };

        resultBox.className = 'form-result is-ok';
        resultBox.innerHTML = '';     // 清空旧内容

        const ok = document.createElement('p');
        ok.textContent = '✅ 校验通过！以下数据将被提交（页面没有刷新）：';
        resultBox.appendChild(ok);

        // 用 definition list（<dl>/<dt>/<dd>）展示键值对
        const dl = document.createElement('dl');
        Object.keys(data).forEach(function (key) {
            const dt = document.createElement('dt');
            dt.textContent = key;
            const dd = document.createElement('dd');
            dd.textContent = data[key];
            dl.appendChild(dt);
            dl.appendChild(dd);
        });
        resultBox.appendChild(dl);

        // 真实项目里这里会发请求：
        // fetch('/api/contact', {
        //     method: 'POST',
        //     headers: { 'Content-Type': 'application/json' },
        //     body: JSON.stringify(data),
        // });

        showToast('表单校验通过（演示：数据未真的发送）');
        console.log('[表单提交]', data);
    }

    // ★★★ submit 事件：必须 preventDefault，否则浏览器会提交表单并刷新整个页面 ★★★
    form.addEventListener('submit', function (event) {
        event.preventDefault();       // 阻止默认的"提交并刷新页面"
        submitForm();
    });

    // 重置按钮：type="reset" 由浏览器负责清空，我们只需要清掉校验状态和结果
    if (resetButton) {
        resetButton.addEventListener('click', function () {
            // 用 setTimeout(…, 0) 让浏览器的 reset 先执行完，再清理我们自己的状态
            window.setTimeout(function () {
                [nameInput, emailInput, messageInput].forEach(function (input) {
                    input.classList.remove('is-valid', 'is-invalid');
                    input.removeAttribute('aria-invalid');
                });
                [nameError, emailError, messageError].forEach(function (el) {
                    if (el) el.textContent = '';
                });
                if (messageCount) {
                    messageCount.textContent = '0 / 200';
                    messageCount.style.color = '';
                }
                resultBox.className = 'form-result';
                resultBox.innerHTML = '<p class="muted">还没有提交任何内容。</p>';
            }, 0);
        });
    }

    /* --- 键盘快捷键：在留言框里按 Ctrl/Cmd + Enter 直接提交 ---------------- */
    messageInput.addEventListener('keydown', function (event) {
        // ctrlKey 是 Windows/Linux 的 Ctrl，metaKey 是 macOS 的 ⌘
        if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
            event.preventDefault();      // 阻止在 textarea 里插入换行
            submitForm();
        }
    });
}


/* =============================================================================
 * 模块 7：零碎功能
 * ========================================================================== */
function initMisc() {
    /* ---- 知识点计数：从 0 数到 42 的小动画 ---- */
    const counter = $('feature-count');
    if (counter) {
        const target = 42;
        const duration = 900;                    // 动画总时长（毫秒）
        const startTime = performance.now();

        /**
         * 每一帧更新一次数字。
         * @param {number} now 当前时间戳（由 requestAnimationFrame 传入）
         */
        function step(now) {
            // 计算进度 0~1，并用 easeOutCubic 让动画"先快后慢"
            const progress = Math.min((now - startTime) / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3);
            counter.textContent = String(Math.round(target * eased));

            if (progress < 1) {
                window.requestAnimationFrame(step);
            } else {
                counter.textContent = String(target);
            }
        }

        window.requestAnimationFrame(step);
    }

    /* ---- 平滑滚动：给所有页内锚点链接加上"考虑页头高度"的滚动 ---- */
    // 说明：CSS 里已经写了 scroll-behavior: smooth，本来不需要 JS。
    // 这里额外用 JS 是为了处理"固定页头遮挡标题"的问题，
    // 顺便演示 preventDefault() 的另一种典型用途（阻止默认的跳转行为）。
    document.querySelectorAll('a[href^="#"]').forEach(function (link) {
        link.addEventListener('click', function (event) {
            const href = link.getAttribute('href');
            if (!href || href === '#') return;

            // querySelector 可以直接用 CSS 选择器语法查找元素
            let target = null;
            try {
                target = document.querySelector(href);
            } catch (err) {
                // href 不是合法的选择器（比如某些特殊字符），忽略即可
                return;
            }
            if (!target) return;

            event.preventDefault();      // 阻止浏览器默认的"瞬间跳转"

            const header = $('site-header');
            const headerHeight = header ? header.offsetHeight : 0;
            // getBoundingClientRect().top 是相对【视口】的位置，
            // 加上 scrollY 就得到相对【文档】的位置。
            const top = target.getBoundingClientRect().top + window.scrollY - headerHeight - 8;

            window.scrollTo({ top: Math.max(top, 0), behavior: 'smooth' });
        });
    });
}


/* =============================================================================
 * 统一入口
 * -----------------------------------------------------------------------------
 * index.html 里的 <script> 已经加了 defer，所以执行到这里时 DOM 一定已经解析完。
 * 但为了"万一有人没写 defer"，这里再判断一次 document.readyState。
 * ========================================================================== */
function initAll() {
    initTheme();
    initMobileMenu();
    initNavHighlight();
    initBackToTop();
    initWorks();
    initContactForm();
    initMisc();
    console.log('[app.js] 个人主页小站的 7 个模块全部初始化完成');
}

if (document.readyState === 'loading') {
    // HTML 还在解析，等 DOMContentLoaded
    document.addEventListener('DOMContentLoaded', initAll);
} else {
    // HTML 已解析完（defer 脚本就是这种情况），直接初始化
    initAll();
}
