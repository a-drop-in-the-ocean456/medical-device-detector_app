// pages/index/index.js
const app = getApp()

Page({
  data: {
    // 图片相关
    imageSrc: '',
    imageBase64: '',
    
    // 检测结果
    detectionResult: null,
    resultImage: '',
    description: '',
    detections: [],
    detectionCount: 0,
    
    // 状态
    isLoading: false,
    hasResult: false,
    errorMsg: '',
    
    // 服务器设置
    serverUrl: '',
    showSettings: false,
    
    // 支持的设备类型提示
    supportedDevices: [
      '药瓶/试剂瓶',
      '医用剪刀',
      '医疗设备键盘',
      '医疗显示器',
      '医疗计时器',
      '医疗设备遥控器',
      '医疗电脑',
      '医疗手册/病历'
    ]
  },

  onLoad() {
    // 页面加载时获取服务器地址
    const savedUrl = wx.getStorageSync('serverUrl')
    const defaultUrl = app.getApiBaseUrl()
    this.setData({
      serverUrl: savedUrl || defaultUrl
    })
    
    // 更新全局配置
    if (savedUrl) {
      app.setApiBaseUrl(savedUrl)
    }
  },

  onShow() {
    // 页面显示时检查网络
    this.checkNetwork()
  },

  // 检查网络状态
  checkNetwork() {
    wx.getNetworkType({
      success: (res) => {
        if (res.networkType === 'none') {
          wx.showToast({
            title: '请检查网络连接',
            icon: 'none',
            duration: 2000
          })
        }
      }
    })
  },

  // 选择图片来源
  chooseImage() {
    wx.showActionSheet({
      itemList: ['拍照', '从相册选择'],
      success: (res) => {
        if (res.tapIndex === 0) {
          this.takePhoto()
        } else if (res.tapIndex === 1) {
          this.selectFromAlbum()
        }
      },
      fail: (res) => {
        console.log('用户取消选择')
      }
    })
  },

  // 拍照
  takePhoto() {
    wx.chooseMedia({
      count: 1,
      mediaType: ['image'],
      sourceType: ['camera'],
      camera: 'back',
      success: (res) => {
        const tempFilePath = res.tempFiles[0].tempFilePath
        this.processSelectedImage(tempFilePath)
      },
      fail: (err) => {
        console.error('拍照失败:', err)
        if (err.errMsg !== 'chooseMedia:fail cancel') {
          wx.showToast({
            title: '拍照失败',
            icon: 'none'
          })
        }
      }
    })
  },

  // 从相册选择
  selectFromAlbum() {
    wx.chooseMedia({
      count: 1,
      mediaType: ['image'],
      sourceType: ['album'],
      success: (res) => {
        const tempFilePath = res.tempFiles[0].tempFilePath
        this.processSelectedImage(tempFilePath)
      },
      fail: (err) => {
        console.error('选择图片失败:', err)
        if (err.errMsg !== 'chooseMedia:fail cancel') {
          wx.showToast({
            title: '选择图片失败',
            icon: 'none'
          })
        }
      }
    })
  },

  // 处理选中的图片
  processSelectedImage(filePath) {
    // 显示图片
    this.setData({
      imageSrc: filePath,
      hasResult: false,
      errorMsg: '',
      detectionResult: null
    })

    // 压缩图片
    wx.compressImage({
      src: filePath,
      quality: 80,
      success: (res) => {
        console.log('图片压缩成功:', res.tempFilePath)
        this.uploadImage(res.tempFilePath)
      },
      fail: (err) => {
        console.error('图片压缩失败:', err)
        // 压缩失败直接使用原图
        this.uploadImage(filePath)
      }
    })
  },

  // 上传图片进行检测
  uploadImage(filePath) {
    const apiUrl = app.getApiBaseUrl()
    
    this.setData({
      isLoading: true,
      errorMsg: ''
    })

    wx.showLoading({
      title: '识别中...',
      mask: true
    })

    // 使用uploadFile上传图片
    wx.uploadFile({
      url: `${apiUrl}/api/detect`,
      filePath: filePath,
      name: 'image',
      timeout: 30000,
      success: (res) => {
        wx.hideLoading()
        
        if (res.statusCode === 200) {
          try {
            const data = JSON.parse(res.data)
            this.handleDetectionResult(data)
          } catch (e) {
            console.error('解析响应失败:', e)
            this.setData({
              isLoading: false,
              errorMsg: '服务器响应格式错误'
            })
          }
        } else {
          console.error('服务器错误:', res.statusCode)
          this.setData({
            isLoading: false,
            errorMsg: `服务器错误 (${res.statusCode})`
          })
          wx.showToast({
            title: '识别失败',
            icon: 'none'
          })
        }
      },
      fail: (err) => {
        wx.hideLoading()
        console.error('上传失败:', err)
        
        let errorMsg = '网络请求失败'
        if (err.errMsg.includes('timeout')) {
          errorMsg = '请求超时，请稍后重试'
        } else if (err.errMsg.includes('fail')) {
          errorMsg = '无法连接到服务器，请检查服务器地址'
        }
        
        this.setData({
          isLoading: false,
          errorMsg: errorMsg
        })
        
        wx.showToast({
          title: errorMsg,
          icon: 'none',
          duration: 2000
        })
      }
    })
  },

  // 处理检测结果
  handleDetectionResult(data) {
    if (data.success) {
      // 解析检测结果
      const detections = data.detections || []
      const description = data.description || ''
      const resultImage = data.image_base64 || ''
      
      this.setData({
        isLoading: false,
        hasResult: true,
        detectionResult: data,
        detections: detections,
        detectionCount: data.detection_count || 0,
        description: description,
        resultImage: resultImage
      })
      
      // 如果有检测结果，显示成功提示
      if (detections.length > 0) {
        wx.showToast({
          title: `识别到${detections.length}个物体`,
          icon: 'success',
          duration: 1500
        })
      } else {
        wx.showToast({
          title: '未识别到设备',
          icon: 'none',
          duration: 1500
        })
      }
    } else {
      this.setData({
        isLoading: false,
        errorMsg: data.error || '识别失败'
      })
      wx.showToast({
        title: data.error || '识别失败',
        icon: 'none'
      })
    }
  },

  // 重新选择图片
  resetImage() {
    this.setData({
      imageSrc: '',
      hasResult: false,
      detectionResult: null,
      resultImage: '',
      description: '',
      detections: [],
      errorMsg: ''
    })
  },

  // 预览结果图片
  previewResultImage() {
    if (this.data.resultImage) {
      wx.previewImage({
        urls: [this.data.resultImage],
        current: this.data.resultImage
      })
    }
  },

  // 预览原始图片
  previewOriginalImage() {
    if (this.data.imageSrc) {
      wx.previewImage({
        urls: [this.data.imageSrc],
        current: this.data.imageSrc
      })
    }
  },

  // 切换设置面板
  toggleSettings() {
    this.setData({
      showSettings: !this.data.showSettings
    })
  },

  // 服务器地址输入
  onServerUrlInput(e) {
    this.setData({
      serverUrl: e.detail.value
    })
  },

  // 保存服务器设置
  saveSettings() {
    const url = this.data.serverUrl.trim()
    
    if (!url) {
      wx.showToast({
        title: '请输入服务器地址',
        icon: 'none'
      })
      return
    }
    
    // 验证URL格式
    if (!url.startsWith('http://') && !url.startsWith('https://')) {
      wx.showToast({
        title: '地址需以http://或https://开头',
        icon: 'none'
      })
      return
    }
    
    // 保存到本地存储
    wx.setStorageSync('serverUrl', url)
    app.setApiBaseUrl(url)
    
    this.setData({
      showSettings: false
    })
    
    wx.showToast({
      title: '设置已保存',
      icon: 'success'
    })
    
    // 测试连接
    this.testConnection()
  },

  // 测试服务器连接
  testConnection() {
    const apiUrl = app.getApiBaseUrl()
    
    wx.request({
      // url: `${apiUrl}/api/health`,
      url: getApp().globalData.apiBase + "/api/health",
      method: 'GET',
      timeout: 5000,
      success: (res) => {
        if (res.statusCode === 200 && res.data.status === 'healthy') {
          wx.showToast({
            title: '服务器连接成功',
            icon: 'success'
          })
        }
      },
      fail: () => {
        wx.showToast({
          title: '服务器连接失败',
          icon: 'none'
        })
      }
    })
  },

  // 复制检测结果
  copyResult() {
    if (!this.data.description) {
      wx.showToast({
        title: '没有可复制的检测结果',
        icon: 'none'
      })
      return
    }
    
    wx.setClipboardData({
      data: this.data.description,
      success: () => {
        wx.showToast({
          title: '已复制到剪贴板',
          icon: 'success'
        })
      }
    })
  },

  // 分享功能
  onShareAppMessage() {
    return {
      title: '医疗设备识别 - 智能识别医疗器材',
      path: '/pages/index/index',
      imageUrl: '/images/share.png'
    }
  },

  // 下拉刷新
  onPullDownRefresh() {
    this.resetImage()
    wx.stopPullDownRefresh()
  },

  // 获取本地IP提示
  showLocalIpTip() {
    wx.showModal({
      title: '如何获取服务器地址',
      content: '1. 在电脑上运行后端服务\n2. 确保手机和电脑在同一WiFi下\n3. 在电脑上打开命令提示符，输入 ipconfig 查看IPv4地址\n4. 格式: http://192.168.x.x:5000',
      showCancel: false,
      confirmText: '知道了'
    })
  }
})

