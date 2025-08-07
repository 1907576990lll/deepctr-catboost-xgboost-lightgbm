export default class Bullet {
  constructor(ctx, x, y, speedX, speedY, owner) {
    this.ctx = ctx
    this.x = x
    this.y = y
    this.width = 4
    this.height = 8
    this.speedX = speedX
    this.speedY = speedY
    this.owner = owner // 'player' 或 'enemy'
  }

  update() {
    this.x += this.speedX
    this.y += this.speedY
  }

  render() {
    if (this.owner === 'player') {
      // 玩家子弹（蓝色光束）
      this.ctx.fillStyle = '#00ffff'
      this.ctx.fillRect(this.x, this.y, this.width, this.height)
      
      // 添加光效
      this.ctx.fillStyle = '#ffffff'
      this.ctx.fillRect(this.x + 1, this.y, this.width - 2, this.height)
      
    } else {
      // 敌机子弹（红色）
      this.ctx.fillStyle = '#ff4444'
      this.ctx.fillRect(this.x, this.y, this.width, this.height)
      
      // 添加暗红色边框
      this.ctx.fillStyle = '#aa0000'
      this.ctx.fillRect(this.x + 1, this.y + 1, this.width - 2, this.height - 2)
    }
  }
}