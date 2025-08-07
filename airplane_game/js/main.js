import Player from './player'
import Enemy from './enemy'
import Bullet from './bullet'
import Background from './background'
import Effects from './effects'
import AudioManager from './audio'
import Config from './config'

export default class Main {
  constructor() {
    this.canvas = window.canvas
    this.ctx = window.context
    
    this.init()
    this.bindEvent()
    this.loop()
  }

  init() {
    // 游戏状态
    this.gameStatus = 'playing' // playing, paused, gameOver
    this.score = 0
    this.level = 1
    this.frameCount = 0
    
    // 游戏对象数组
    this.enemies = []
    this.playerBullets = []
    this.enemyBullets = []
    
    // 创建游戏对象
    this.background = new Background(this.ctx, this.canvas.width, this.canvas.height)
    this.player = new Player(this.ctx, this.canvas.width / 2, this.canvas.height - 100)
    this.effects = new Effects(this.ctx)
    this.audio = new AudioManager()
    
    // 开始背景音乐
    this.audio.playBackgroundMusic()
    
    // 敌机生成间隔
    this.enemySpawnInterval = Config.enemy.spawnInterval
    this.lastEnemySpawn = 0
    this.currentLevel = 0
    this.difficultyMultiplier = 1.0
  }

  bindEvent() {
    // 监听触摸事件
    wx.onTouchStart(this.onTouchStart.bind(this))
    wx.onTouchMove(this.onTouchMove.bind(this))
    wx.onTouchEnd(this.onTouchEnd.bind(this))
  }

  onTouchStart(e) {
    if (this.gameStatus !== 'playing') return
    
    const touch = e.touches[0]
    this.player.startMove(touch.clientX, touch.clientY)
  }

  onTouchMove(e) {
    if (this.gameStatus !== 'playing') return
    
    const touch = e.touches[0]
    this.player.move(touch.clientX, touch.clientY)
  }

  onTouchEnd(e) {
    if (this.gameStatus !== 'playing') return
    this.player.stopMove()
  }

  update() {
    if (this.gameStatus !== 'playing') return
    
    this.frameCount++
    
    // 更新背景
    this.background.update()
    
    // 更新玩家
    this.player.update()
    
    // 玩家自动射击
    if (this.frameCount % Config.player.shootInterval === 0) {
      const bullet = this.player.shoot()
      if (bullet) {
        this.playerBullets.push(bullet)
        this.audio.playShoot()
      }
    }
    
    // 生成敌机
    if (this.frameCount - this.lastEnemySpawn > this.enemySpawnInterval) {
      this.spawnEnemy()
      this.lastEnemySpawn = this.frameCount
    }
    
    // 更新子弹
    this.updateBullets()
    
    // 更新敌机
    this.updateEnemies()
    
    // 更新特效
    this.effects.update()
    
    // 检查关卡进度
    this.checkLevelProgress()
    
    // 碰撞检测
    this.checkCollisions()
    
    // 清理超出屏幕的对象
    this.cleanup()
  }

  updateBullets() {
    // 更新玩家子弹
    this.playerBullets.forEach(bullet => bullet.update())
    
    // 更新敌机子弹
    this.enemyBullets.forEach(bullet => bullet.update())
  }

  updateEnemies() {
    this.enemies.forEach(enemy => {
      enemy.update()
      
      // 敌机射击
      if (Math.random() < 0.01) {
        const bullet = enemy.shoot()
        if (bullet) {
          this.enemyBullets.push(bullet)
        }
      }
    })
  }

  spawnEnemy() {
    const x = Math.random() * (this.canvas.width - 60) + 30
    const enemy = new Enemy(this.ctx, x, -50)
    this.enemies.push(enemy)
  }

  checkCollisions() {
    // 玩家子弹击中敌机
    this.playerBullets.forEach((bullet, bulletIndex) => {
      this.enemies.forEach((enemy, enemyIndex) => {
                  if (this.isColliding(bullet, enemy)) {
            // 创建爆炸效果
            this.effects.createExplosion(enemy.x + enemy.width / 2, enemy.y + enemy.height / 2, 'normal')
            
            // 播放击中音效
            this.audio.playHit()
            
            // 移除子弹和敌机
            this.playerBullets.splice(bulletIndex, 1)
            this.enemies.splice(enemyIndex, 1)
            
            // 增加分数
            const points = enemy.type === 'fast' ? Config.enemy.fast.points : Config.enemy.normal.points
            this.score += Math.floor(points * this.difficultyMultiplier)
          }
      })
    })
    
    // 敌机子弹击中玩家
    this.enemyBullets.forEach((bullet, bulletIndex) => {
      if (this.isColliding(bullet, this.player)) {
        this.effects.createExplosion(this.player.x + this.player.width / 2, this.player.y + this.player.height / 2, 'big')
        this.audio.playExplosion()
        this.gameOver()
      }
    })
    
    // 敌机撞击玩家
    this.enemies.forEach(enemy => {
      if (this.isColliding(enemy, this.player)) {
        this.effects.createExplosion(this.player.x + this.player.width / 2, this.player.y + this.player.height / 2, 'big')
        this.audio.playExplosion()
        this.gameOver()
      }
    })
  }

  isColliding(obj1, obj2) {
    return obj1.x < obj2.x + obj2.width &&
           obj1.x + obj1.width > obj2.x &&
           obj1.y < obj2.y + obj2.height &&
           obj1.y + obj1.height > obj2.y
  }

  checkLevelProgress() {
    // 检查是否应该提升难度
    const nextLevel = Config.levels.find(level => 
      level.score <= this.score && 
      level.score > (Config.levels[this.currentLevel] ? Config.levels[this.currentLevel].score : -1)
    )
    
    if (nextLevel && Config.levels.indexOf(nextLevel) > this.currentLevel) {
      this.currentLevel = Config.levels.indexOf(nextLevel)
      this.level = this.currentLevel + 1
      this.enemySpawnInterval = nextLevel.enemySpawnInterval
      this.difficultyMultiplier = nextLevel.difficultyMultiplier
      
      // 显示关卡提升提示
      this.showLevelUpMessage()
    }
  }

  showLevelUpMessage() {
    // 这里可以添加关卡提升的视觉提示
    // 暂时只是一个简单的标记
    this.levelUpMessage = {
      text: `等级 ${this.level}!`,
      time: 120 // 显示120帧
    }
  }

  cleanup() {
    // 清理超出屏幕的子弹
    this.playerBullets = this.playerBullets.filter(bullet => 
      bullet.y > -bullet.height
    )
    
    this.enemyBullets = this.enemyBullets.filter(bullet => 
      bullet.y < this.canvas.height + bullet.height
    )
    
    // 清理超出屏幕的敌机
    this.enemies = this.enemies.filter(enemy => 
      enemy.y < this.canvas.height + enemy.height
    )
  }

  render() {
    // 清空画布
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height)
    
    // 渲染背景
    this.background.render()
    
    // 渲染玩家
    this.player.render()
    
    // 渲染子弹
    this.playerBullets.forEach(bullet => bullet.render())
    this.enemyBullets.forEach(bullet => bullet.render())
    
    // 渲染敌机
    this.enemies.forEach(enemy => enemy.render())
    
    // 渲染特效
    this.effects.render()
    
    // 渲染UI
    this.renderUI()
  }

  renderUI() {
    this.ctx.fillStyle = '#fff'
    this.ctx.font = '20px Arial'
    this.ctx.fillText(`分数: ${this.score}`, 20, 40)
    this.ctx.fillText(`等级: ${this.level}`, 20, 70)
    
    // 显示关卡提升消息
    if (this.levelUpMessage && this.levelUpMessage.time > 0) {
      this.ctx.fillStyle = '#ffff00'
      this.ctx.font = '24px Arial'
      this.ctx.textAlign = 'center'
      this.ctx.fillText(this.levelUpMessage.text, this.canvas.width / 2, this.canvas.height / 3)
      this.ctx.textAlign = 'left'
      this.levelUpMessage.time--
      
      if (this.levelUpMessage.time <= 0) {
        this.levelUpMessage = null
      }
    }
    
    if (this.gameStatus === 'gameOver') {
      this.ctx.fillStyle = 'rgba(0, 0, 0, 0.7)'
      this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height)
      
      this.ctx.fillStyle = '#fff'
      this.ctx.font = '30px Arial'
      this.ctx.textAlign = 'center'
      this.ctx.fillText('游戏结束', this.canvas.width / 2, this.canvas.height / 2 - 50)
      this.ctx.fillText(`最终分数: ${this.score}`, this.canvas.width / 2, this.canvas.height / 2)
      this.ctx.fillText('点击重新开始', this.canvas.width / 2, this.canvas.height / 2 + 50)
      this.ctx.textAlign = 'left'
    }
  }

  gameOver() {
    this.gameStatus = 'gameOver'
    
    // 播放游戏结束音效
    this.audio.playGameOver()
    this.audio.stopBackgroundMusic()
    
    // 监听重新开始
    const restart = () => {
      this.init()
      this.gameStatus = 'playing'
      this.audio.playBackgroundMusic()
      wx.offTouchStart(restart)
    }
    wx.onTouchStart(restart)
  }

  loop() {
    this.update()
    this.render()
    
    requestAnimationFrame(this.loop.bind(this))
  }
}