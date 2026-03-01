// app.js - 小程序入口文件
App({
  globalData: {
    // 后端API地址
    // 开发时使用本地IP，生产环境替换为实际域名
    // 手机测试时需要使用电脑的局域网IP地址
    // apiBaseUrl: 'http://192.168.16.102:5000',
    // apiBaseUrl: 'http://106.14.16.19:5000',
    apiBase: "https://m-r.asia",
    
    // 备用：如果使用ngrok内网穿透
    // apiBaseUrl: 'https://your-ngrok-url.ngrok.io',
    
    userInfo: null
  },

  onLaunch() {
    // 小程序启动时的初始化
    console.log('Medical Device Detector App Launched')
    
    // 检查网络状态
    wx.getNetworkType({
      success: (res) => {
        console.log('Network type:', res.networkType)
        if (res.networkType === 'none') {
          wx.showToast({
            title: '请检查网络连接',
            icon: 'none',
            duration: 2000
          })
        }
      }
    })
    
    // 监听网络状态变化
    wx.onNetworkStatusChange((res) => {
      console.log('Network status changed:', res.isConnected, res.networkType)
      if (!res.isConnected) {
        wx.showToast({
          title: '网络已断开',
          icon: 'none'
        })
      }
    })
  },

  onShow() {
    console.log('App shown')
  },

  onHide() {
    console.log('App hidden')
  },

  // 全局错误处理
  onError(msg) {
    console.error('App error:', msg)
  },

  // 获取API基础URL
  getApiBaseUrl() {
    return this.globalData.apiBaseUrl
  },

  // 设置API基础URL（用于动态切换）
  setApiBaseUrl(url) {
    this.globalData.apiBaseUrl = url
    console.log('API base URL updated:', url)
  }
})
