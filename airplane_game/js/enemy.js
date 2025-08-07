import Bullet from './bullet'

export default class Enemy {
  constructor(ctx, x, y) {
    this.ctx = ctx
    this.x = x
    this.y = y
    this.width = 50
    this.height = 60
    
    // 移动速度
    this.speedX = (Math.random() - 0.5) * 2 // -1 到 1 的随机速度
    this.speedY = 2 + Math.random() * 2    // 2 到 4 的随机速度
    
    // 射击相关
    this.lastShoot = 0
    this.shootInterval = 60 + Math.random() * 60 // 60-120帧射击一次
    
    // 生命值
    this.hp = 1
    
    // 敌机类型
    this.type = Math.random() < 0.8 ? 'normal' : 'fast'
    if (this.type === 'fast') {
      this.speedY *= 1.5
      this.width = 40
      this.height = 50
    }
  }

  update() {
    // 移动
    this.x += this.speedX
    this.y += this.speedY
    
    // 边界反弹
    if (this.x <= 0 || this.x >= this.ctx.canvas.width - this.width) {
      this.speedX = -this.speedX
    }
    
    // 限制在屏幕内
    this.x = Math.max(0, Math.min(this.x, this.ctx.canvas.width - this.width))
  }

  shoot() {
    const currentFrame = Date.now()
    if (currentFrame - this.lastShoot < this.shootInterval * 16) {
      return null
    }
    
    this.lastShoot = currentFrame
    
    // 创建子弹
    const bulletX = this.x + this.width / 2 - 2
    const bulletY = this.y + this.height
    
    return new Bullet(this.ctx, bulletX, bulletY, 0, 5, 'enemy')
  }

  render() {
    if (this.type === 'normal') {
      // 普通敌机（红色）
      this.ctx.fillStyle = '#ff0000'
      this.ctx.beginPath()
      
      // 机身
      this.ctx.moveTo(this.x + this.width / 2, this.y + this.height)
      this.ctx.lineTo(this.x, this.y)
      this.ctx.lineTo(this.x + this.width, this.y)
      this.ctx.closePath()
      this.ctx.fill()
      
      // 机翼
      this.ctx.fillStyle = '#cc0000'
      this.ctx.fillRect(this.x + 5, this.y + this.height * 0.3, 12, 6)
      this.ctx.fillRect(this.x + this.width - 17, this.y + this.height * 0.3, 12, 6)
      
    } else {
      // 快速敌机（紫色）
      this.ctx.fillStyle = '#8800ff'
      this.ctx.beginPath()
      
      // 更尖锐的机身
      this.ctx.moveTo(this.x + this.width / 2, this.y + this.height)
      this.ctx.lineTo(this.x + this.width * 0.2, this.y)
      this.ctx.lineTo(this.x + this.width * 0.8, this.y)
      this.ctx.closePath()
      this.ctx.fill()
      
      // 小机翼
      this.ctx.fillStyle = '#6600cc'
      this.ctx.fillRect(this.x + 3, this.y + this.height * 0.4, 8, 4)
      this.ctx.fillRect(this.x + this.width - 11, this.y + this.height * 0.4, 8, 4)
    }
    
    // 驾驶舱
    this.ctx.fillStyle = '#ffff00'
    this.ctx.beginPath()
    this.ctx.arc(this.x + this.width / 2, this.y + this.height * 0.6, 4, 0, Math.PI * 2)
    this.ctx.fill()
  }
}