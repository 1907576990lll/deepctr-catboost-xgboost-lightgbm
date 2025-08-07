// Symbol 兼容性处理
if (!window.Symbol) {
  window.Symbol = {}
  window.Symbol.iterator = '@@iterator'
}