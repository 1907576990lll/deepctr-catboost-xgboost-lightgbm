export default class AudioManager {
  constructor() {
    this.sounds = {}
    this.backgroundMusic = null
    this.isMuted = false
    
    // 初始化音效（使用音频上下文生成简单音效）
    this.initAudioContext()
  }

  initAudioContext() {
    try {
      // 微信小游戏环境下的音频处理
      this.audioContext = wx.createWebAudioContext()
    } catch (e) {
      console.log('AudioContext not supported')
      this.audioContext = null
    }
  }

  // 创建简单的音效
  createTone(frequency, duration, type = 'sine') {
    if (!this.audioContext || this.isMuted) return

    const oscillator = this.audioContext.createOscillator()
    const gainNode = this.audioContext.createGain()
    
    oscillator.connect(gainNode)
    gainNode.connect(this.audioContext.destination)
    
    oscillator.frequency.value = frequency
    oscillator.type = type
    
    gainNode.gain.setValueAtTime(0.1, this.audioContext.currentTime)
    gainNode.gain.exponentialRampToValueAtTime(0.01, this.audioContext.currentTime + duration)
    
    oscillator.start(this.audioContext.currentTime)
    oscillator.stop(this.audioContext.currentTime + duration)
  }

  // 播放射击音效
  playShoot() {
    this.createTone(800, 0.1, 'square')
  }

  // 播放爆炸音效
  playExplosion() {
    if (!this.audioContext || this.isMuted) return
    
    // 复杂的爆炸音效
    const noiseBuffer = this.createNoiseBuffer(0.3)
    const source = this.audioContext.createBufferSource()
    const filter = this.audioContext.createBiquadFilter()
    const gainNode = this.audioContext.createGain()
    
    source.buffer = noiseBuffer
    source.connect(filter)
    filter.connect(gainNode)
    gainNode.connect(this.audioContext.destination)
    
    filter.type = 'lowpass'
    filter.frequency.setValueAtTime(2000, this.audioContext.currentTime)
    filter.frequency.exponentialRampToValueAtTime(100, this.audioContext.currentTime + 0.3)
    
    gainNode.gain.setValueAtTime(0.3, this.audioContext.currentTime)
    gainNode.gain.exponentialRampToValueAtTime(0.01, this.audioContext.currentTime + 0.3)
    
    source.start(this.audioContext.currentTime)
  }

  // 播放敌机被击中音效
  playHit() {
    this.createTone(300, 0.15, 'sawtooth')
  }

  // 播放游戏结束音效
  playGameOver() {
    // 下降音调的音效
    if (!this.audioContext || this.isMuted) return
    
    const oscillator = this.audioContext.createOscillator()
    const gainNode = this.audioContext.createGain()
    
    oscillator.connect(gainNode)
    gainNode.connect(this.audioContext.destination)
    
    oscillator.frequency.setValueAtTime(400, this.audioContext.currentTime)
    oscillator.frequency.exponentialRampToValueAtTime(100, this.audioContext.currentTime + 1)
    oscillator.type = 'triangle'
    
    gainNode.gain.setValueAtTime(0.2, this.audioContext.currentTime)
    gainNode.gain.exponentialRampToValueAtTime(0.01, this.audioContext.currentTime + 1)
    
    oscillator.start(this.audioContext.currentTime)
    oscillator.stop(this.audioContext.currentTime + 1)
  }

  // 创建噪音缓冲区（用于爆炸音效）
  createNoiseBuffer(duration) {
    if (!this.audioContext) return null
    
    const frameCount = this.audioContext.sampleRate * duration
    const buffer = this.audioContext.createBuffer(1, frameCount, this.audioContext.sampleRate)
    const output = buffer.getChannelData(0)
    
    for (let i = 0; i < frameCount; i++) {
      output[i] = Math.random() * 2 - 1
    }
    
    return buffer
  }

  // 播放背景音乐（简单的循环旋律）
  playBackgroundMusic() {
    if (!this.audioContext || this.isMuted) return
    
    this.stopBackgroundMusic()
    
    const melody = [220, 247, 262, 294, 330, 294, 262, 247] // 简单旋律
    let noteIndex = 0
    
    const playNote = () => {
      if (this.isMuted) return
      
      const frequency = melody[noteIndex]
      this.createTone(frequency, 0.5, 'sine')
      
      noteIndex = (noteIndex + 1) % melody.length
    }
    
    // 每0.6秒播放一个音符
    this.musicInterval = setInterval(playNote, 600)
  }

  // 停止背景音乐
  stopBackgroundMusic() {
    if (this.musicInterval) {
      clearInterval(this.musicInterval)
      this.musicInterval = null
    }
  }

  // 切换静音状态
  toggleMute() {
    this.isMuted = !this.isMuted
    if (this.isMuted) {
      this.stopBackgroundMusic()
    } else {
      this.playBackgroundMusic()
    }
    return this.isMuted
  }

  // 设置音量
  setVolume(volume) {
    this.volume = Math.max(0, Math.min(1, volume))
  }
}