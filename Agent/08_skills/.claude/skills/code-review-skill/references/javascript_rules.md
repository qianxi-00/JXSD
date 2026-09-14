# JavaScript / TypeScript 代码审查规范（30 条）

> 本文件只在审查 JavaScript / TypeScript 代码时才需要读入上下文。

## 一、变量与作用域

1. 默认用 `const`；需要重新赋值才用 `let`；永远不要用 `var`。
2. 用 `===` / `!==`，不要用 `==` / `!=`（隐式类型转换是 bug 温床）。
3. 变量声明放在使用点附近，不要把所有声明堆在函数开头。
4. 避免在块级作用域外泄漏变量（尤其 `for` 循环里的闭包）。

## 二、函数与异步

5. 优先箭头函数；需要 `this` 动态绑定时才用 `function`。
6. 参数超过 3 个时改用「选项对象」解构。
7. `async` 函数里每个 `await` 都要能抛出可处理的错误。
8. 不要在 `forEach` 里 `await` —— 用 `for...of`，否则不会等待。
9. 并发请求用 `Promise.all` / `allSettled`，不要串行 `await` 拖慢。
10. 永远不要忘记 `catch`；`Promise` 链结尾必须有 `.catch()`。

## 三、错误处理

11. `throw` 只抛 `Error` 及其子类，不要抛字符串。
12. 自定义错误继承 `Error` 并设置 `this.name`，否则堆栈里认不出来。
13. 顶层用 try/catch 包住事件回调，防止一个异常打断整个流程。

## 四、性能

14. 循环内不要做 DOM 查询；先缓存到变量。
15. 频繁触发的 `scroll` / `resize` / `input` 要防抖（debounce）或节流（throttle）。
16. 大数组用 `map` / `filter` / `reduce`，但不要串成五层链式调用 —— 可读性优先。
17. 避免在渲染函数里创建新对象/新函数（会破坏 memo 优化）。

## 五、安全

18. 不要用 `innerHTML` 渲染外部数据；用 `textContent` 或框架的转义机制。
19. 不要 `eval` / `new Function`。
20. `postMessage` 必须校验 `event.origin`。
21. 敏感 token 不要放 `localStorage`，优先 httpOnly Cookie。

## 六、TypeScript 专项

22. 禁止 `any`；实在不确定用 `unknown` 再收窄。
23. 接口用 `interface`，联合/交叉类型用 `type`。
24. 函数返回值显式标注，尤其是导出函数。
25. 用可选链 `?.` 和空值合并 `??`，不要写 `a && a.b && a.b.c`。

## 七、工程化

26. 文件不超过 400 行；超了就按职责拆分。
27. 一个文件只做一件事；`index.js` 只做再导出。
28. 依赖锁文件（`package-lock.json` / `pnpm-lock.yaml`）必须提交。
29. 不要提交 `console.log`；用统一的 logger。
30. 涉及金额、时间、ID 的运算，注意浮点精度与字符串/数字混用。
