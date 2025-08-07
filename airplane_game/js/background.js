export default class Background {
  constructor(ctx, width, height) {
    this.ctx = ctx
    this.width = width
    this.height = height
    
    // 星星数组
    this.stars = []
    this.initStars()
    
    // 背景渐变偏移
    this.gradientOffset = 0
  }

  initStars() {
    // 创建随机星星
    for (let i = 0; i < 100; i++) {
      this.stars.push({
        x: Math.random() * this.width,
        y: Math.random() * this.height,
        size: Math.random() * 2 + 1,
        speed: Math.random() * 2 + 1,
        opacity: Math.random() * 0.8 + 0.2
      })
    }
  }

  update() {
    // 更新星星位置
    this.stars.forEach(star => {
      star.y += star.speed
      
      // 如果星星移出屏幕底部，重新从顶部出现
      if (star.y > this.height) {
        star.y = -star.size
        star.x = Math.random() * this.width
      }
    })
    
    // 更新背景渐变
    this.gradientOffset += 0.5
    if (this.gradientOffset >= this.height) {
      this.gradientOffset = 0
    }
  }

  render() {
    // 绘制深空背景渐变
    const gradient = this.ctx.createLinearGradient(0, 0, 0, this.height)
    gradient.addColorStop(0, '#000033')
    gradient.addColorStop(0.5, '#000066')
    gradient.addColorStop(1, '#000099')
    
    this.ctx.fillStyle = gradient
    this.ctx.fillRect(0, 0, this.width, this.height)
    
    // 绘制星星
    this.stars.forEach(star => {
      this.ctx.fillStyle = `rgba(255, 255, 255, ${star.opacity})`
      this.ctx.beginPath()
      this.ctx.arc(star.x, star.y, star.size, 0, Math.PI * 2)
      this.ctx.fill()
      
      // 添加星星闪烁效果
      if (Math.random() < 0.1) {
        this.ctx.fillStyle = `rgba(255, 255, 255, ${star.opacity * 0.5})`
        this.ctx.beginPath()
        this.ctx.arc(star.x, star.y, star.size + 2, 0, Math.PI * 2)
        this.ctx.fill()
      }
    })
    
    // 绘制一些远景行星
    this.renderPlanets()
  }

  renderPlanets() {
    // 绘制远景装饰元素
    this.ctx.fillStyle = 'rgba(100, 100, 200, 0.3)'
    this.ctx.beginPath()
    this.ctx.arc(this.width * 0.8, this.height * 0.2, 20, 0, Math.PI * 2)
    this.ctx.fill()
    
    this.ctx.fillStyle = 'rgba(200, 100, 100, 0.2)'
    this.ctx.beginPath()
    this.ctx.arc(this.width * 0.1, this.height * 0.7, 15, 0, Math.PI * 2)
    this.ctx.fill()
  }
}