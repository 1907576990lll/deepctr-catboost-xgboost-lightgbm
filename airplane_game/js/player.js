import Bullet from './bullet'

export default class Player {
  constructor(ctx, x, y) {
    this.ctx = ctx
    this.x = x
    this.y = y
    this.width = 60
    this.height = 80
    
    // 移动相关
    this.isMoving = false
    this.targetX = x
    this.targetY = y
    this.speed = 8
    
    // 射击相关
    this.lastShoot = 0
    this.shootInterval = 8 // 射击间隔帧数
  }

  startMove(x, y) {
    this.isMoving = true
    this.targetX = x - this.width / 2
    this.targetY = y - this.height / 2
    
    // 限制移动范围
    this.targetX = Math.max(0, Math.min(this.targetX, this.ctx.canvas.width - this.width))
    this.targetY = Math.max(0, Math.min(this.targetY, this.ctx.canvas.height - this.height))
  }

  move(x, y) {
    if (!this.isMoving) return
    
    this.targetX = x - this.width / 2
    this.targetY = y - this.height / 2
    
    // 限制移动范围
    this.targetX = Math.max(0, Math.min(this.targetX, this.ctx.canvas.width - this.width))
    this.targetY = Math.max(0, Math.min(this.targetY, this.ctx.canvas.height - this.height))
  }

  stopMove() {
    this.isMoving = false
  }

  update() {
    if (this.isMoving) {
      // 平滑移动到目标位置
      const dx = this.targetX - this.x
      const dy = this.targetY - this.y
      
      this.x += dx * 0.2
      this.y += dy * 0.2
    }
  }

  shoot() {
    const currentFrame = Date.now()
    if (currentFrame - this.lastShoot < this.shootInterval * 16) { // 约16ms一帧
      return null
    }
    
    this.lastShoot = currentFrame
    
    // 创建子弹
    const bulletX = this.x + this.width / 2 - 2
    const bulletY = this.y
    
    return new Bullet(this.ctx, bulletX, bulletY, 0, -10, 'player')
  }

  render() {
    // 绘制玩家飞机（简单的三角形）
    this.ctx.fillStyle = '#00ff00'
    this.ctx.beginPath()
    
    // 机身
    this.ctx.moveTo(this.x + this.width / 2, this.y)
    this.ctx.lineTo(this.x, this.y + this.height)
    this.ctx.lineTo(this.x + this.width, this.y + this.height)
    this.ctx.closePath()
    this.ctx.fill()
    
    // 机翼
    this.ctx.fillStyle = '#00cc00'
    this.ctx.fillRect(this.x + 5, this.y + this.height * 0.6, 15, 8)
    this.ctx.fillRect(this.x + this.width - 20, this.y + this.height * 0.6, 15, 8)
    
    // 驾驶舱
    this.ctx.fillStyle = '#0088ff'
    this.ctx.beginPath()
    this.ctx.arc(this.x + this.width / 2, this.y + this.height * 0.4, 8, 0, Math.PI * 2)
    this.ctx.fill()
  }
}