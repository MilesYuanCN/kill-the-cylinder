import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "Kill-The-Cylinder",
  description: "Kill-The-Cylinder",
  
  head: [
    ['link', { rel: 'stylesheet', href: 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css' }],
    ['script', { defer: 'true', src: 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js' }],
    ['script', { defer: 'true', src: 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js' }],
    ['script', { 
      defer: 'true',
      content: `
        document.addEventListener("DOMContentLoaded", function() {
          renderMathInElement(document.body, {
            delimiters: [
              {left: '$$', right: '$$', display: true},
              {left: '$', right: '$', display: false},
              {left: '\\(', right: '\\)', display: false},
              {left: '\\[', right: '\\]', display: true}
            ],
            throwOnError : false
          });
        });
      `
    }]
  ],

  themeConfig: {
    // 禁用导航栏
    nav: [],

    // 禁用侧边栏
    sidebar: [],

    // 禁用社交链接
    socialLinks: [],

    // 禁用页脚
    footer: {
      message: '',
      copyright: ''
    }
  },

  markdown: {
    math: true
  }
})
