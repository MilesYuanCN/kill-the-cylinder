import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "Kill-The-Cylinder",
  description: "Kill-The-Cylinder",
  
  // 添加首页重定向
  rewrites: {
    'Learn_RL_From_DeepSeek_0.md': 'index.md'
  },

  themeConfig: {
    // 禁用导航栏
    nav: false,

    // 禁用侧边栏
    sidebar: false,

    // 禁用社交链接
    socialLinks: false,

    // 禁用页脚
    footer: false
  }
})
