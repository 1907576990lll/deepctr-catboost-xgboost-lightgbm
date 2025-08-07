export default class Effects {
  constructor(ctx) {
    this.ctx = ctx
    this.explosions = []
    this.particles = []
  }

  createExplosion(x, y, type = 'normal') {
    const explosion = {
      x: x,
      y: y,
      radius: 0,
      maxRadius: type === 'big' ? 50 : 30,
      life: 0,
      maxLife: 20,
      type: type
    }
    this.explosions.push(explosion)
    
    // 创建爆炸粒子
    for (let i = 0; i < (type === 'big' ? 15 : 8); i++) {
      this.createParticle(x, y, type)
    }
  }

  createParticle(x, y, type) {
    const particle = {
      x: x,
      y: y,
      vx: (Math.random() - 0.5) * 10,
      vy: (Math.random() - 0.5) * 10,
      life: 0,
      maxLife: 15 + Math.random() * 10,
      size: Math.random() * 4 + 2,
      color: type === 'big' ? 'orange' : 'red'
    }
    this.particles.push(particle)
  }

  update() {
    // 更新爆炸效果
    this.explosions = this.explosions.filter(explosion => {
      explosion.life++
      explosion.radius = (explosion.life / explosion.maxLife) * explosion.maxRadius
      return explosion.life < explosion.maxLife
    })

    // 更新粒子效果
    this.particles = this.particles.filter(particle => {
      particle.life++
      particle.x += particle.vx
      particle.y += particle.vy
      particle.vx *= 0.98
      particle.vy *= 0.98
      particle.size *= 0.95
      return particle.life < particle.maxLife && particle.size > 0.5
    })
  }

  render() {
    // 渲染爆炸效果
    this.explosions.forEach(explosion => {
      const alpha = 1 - (explosion.life / explosion.maxLife)
      const gradient = this.ctx.createRadialGradient(
        explosion.x, explosion.y, 0,
        explosion.x, explosion.y, explosion.radius
      )
      
      if (explosion.type === 'big') {
        gradient.addColorStop(0, `rgba(255, 255, 0, ${alpha})`)
        gradient.addColorStop(0.5, `rgba(255, 100, 0, ${alpha * 0.8})`)
        gradient.addColorStop(1, `rgba(255, 0, 0, 0)`)
      } else {
        gradient.addColorStop(0, `rgba(255, 200, 0, ${alpha})`)
        gradient.addColorStop(0.7, `rgba(255, 0, 0, ${alpha * 0.6})`)
        gradient.addColorStop(1, `rgba(100, 0, 0, 0)`)
      }
      
      this.ctx.fillStyle = gradient
      this.ctx.beginPath()
      this.ctx.arc(explosion.x, explosion.y, explosion.radius, 0, Math.PI * 2)
      this.ctx.fill()
    })

    // 渲染粒子效果
    this.particles.forEach(particle => {
      const alpha = 1 - (particle.life / particle.maxLife)
      this.ctx.fillStyle = particle.color === 'orange' 
        ? `rgba(255, 165, 0, ${alpha})`
        : `rgba(255, 0, 0, ${alpha})`
      
      this.ctx.beginPath()
      this.ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2)
      this.ctx.fill()
    })
  }
}