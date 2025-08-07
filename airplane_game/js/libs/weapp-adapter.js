// 微信小游戏适配器
const canvas = wx.createCanvas()
const context = canvas.getContext('2d')

// 获取系统信息
const systemInfo = wx.getSystemInfoSync()
canvas.width = systemInfo.windowWidth
canvas.height = systemInfo.windowHeight

// 导出全局对象
window.canvas = canvas
window.context = context
window.devicePixelRatio = systemInfo.pixelRatio

// Image 构造函数
window.Image = function() {
  const image = wx.createImage()
  return image
}

// Audio 构造函数  
window.Audio = function() {
  const audio = wx.createInnerAudioContext()
  return audio
}

// 禁用多点触控
canvas.addEventListener = () => {}
canvas.removeEventListener = () => {}

// 添加requestAnimationFrame支持
if (!window.requestAnimationFrame) {
  window.requestAnimationFrame = callback => {
    return setTimeout(callback, 1000/60)
  }
}

if (!window.cancelAnimationFrame) {
  window.cancelAnimationFrame = clearTimeout
}