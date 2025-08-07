export default {
  // 游戏基础设置
  game: {
    fps: 60,
    width: 375,  // 默认宽度
    height: 667, // 默认高度
  },
  
  // 玩家设置
  player: {
    width: 60,
    height: 80,
    speed: 8,
    shootInterval: 10, // 射击间隔（帧）
    lives: 3,
  },
  
  // 敌机设置
  enemy: {
    spawnInterval: 60, // 生成间隔（帧）
    spawnIntervalMin: 30, // 最小生成间隔
    normal: {
      width: 50,
      height: 60,
      speed: 2,
      speedVariation: 2,
      points: 10,
      probability: 0.8
    },
    fast: {
      width: 40,
      height: 50,
      speed: 3,
      speedVariation: 2,
      points: 20,
      probability: 0.2
    }
  },
  
  // 子弹设置
  bullet: {
    player: {
      width: 4,
      height: 8,
      speed: 10,
      color: '#00ffff'
    },
    enemy: {
      width: 4,
      height: 8,
      speed: 5,
      color: '#ff4444'
    }
  },
  
  // 关卡设置
  levels: [
    { score: 0, enemySpawnInterval: 60, difficultyMultiplier: 1.0 },
    { score: 200, enemySpawnInterval: 50, difficultyMultiplier: 1.2 },
    { score: 500, enemySpawnInterval: 40, difficultyMultiplier: 1.4 },
    { score: 1000, enemySpawnInterval: 35, difficultyMultiplier: 1.6 },
    { score: 2000, enemySpawnInterval: 30, difficultyMultiplier: 1.8 },
  ],
  
  // 视觉效果设置
  effects: {
    explosion: {
      normal: { maxRadius: 30, life: 20, particles: 8 },
      big: { maxRadius: 50, life: 25, particles: 15 }
    },
    particle: {
      life: 25,
      sizeVariation: 4,
      speedVariation: 10
    }
  },
  
  // 音频设置
  audio: {
    backgroundMusic: true,
    soundEffects: true,
    volume: 0.7
  },
  
  // 背景设置
  background: {
    starCount: 100,
    starSpeedMin: 1,
    starSpeedMax: 3,
    starSizeMin: 1,
    starSizeMax: 3
  }
}