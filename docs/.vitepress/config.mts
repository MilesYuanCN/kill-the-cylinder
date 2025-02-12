import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "Kill-The-Cylinder",
  description: "Kill-The-Cylinder",
  
  // 添加首页重定向
  rewrites: {
    // 'index.md': 'Learn_RL_From_DeepSeek_0.md'
    'Learn_RL_From_DeepSeek_0.md': 'index.md'
  },

  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: 'Home', link: '/Learn_RL_From_DeepSeek_0' },
      { text: 'Examples', link: '/markdown-examples' }
    ],

    sidebar: [
      {
        text: 'Learn RL',
        items: [
          { text: 'Learn RL From DeepSeek', link: '/Learn_RL_From_DeepSeek_0' },
          { text: 'Markdown Examples', link: '/markdown-examples' },
          { text: 'Runtime API Examples', link: '/api-examples' }
        ]
      }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/vuejs/vitepress' }
    ]
  }
})
