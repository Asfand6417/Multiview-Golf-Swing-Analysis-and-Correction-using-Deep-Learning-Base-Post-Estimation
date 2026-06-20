/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        surface: {
          base: '#131313',
          dim: '#131313',
          bright: '#3a3939',
          lowest: '#0e0e0e',
          low: '#1c1b1b',
          container: '#201f1f',
          high: '#2a2a2a',
          highest: '#353534',
        },
        brand: {
          primary: '#75ff9e',
          'primary-dim': '#00e475',
          'on-primary': '#003918',
          'primary-container': '#00e676',
          'on-primary-container': '#00612e',
          secondary: '#c6c6c9',
          'on-secondary': '#2f3133',
          tertiary: '#dae4ff',
          'on-tertiary': '#002e6a',
          'tertiary-container': '#b0c8ff',
          error: '#ffb4ab',
          'on-error': '#690005',
          warning: '#fbcfe8',
          amber: '#f59e0b',
          blue: '#00e4ff',
        },
        text: {
          main: '#e5e2e1',
          muted: '#bacbb9',
          dim: '#859585',
        },
        border: {
          DEFAULT: '#3b4a3d',
        },
        'on-primary': '#00210b',
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['Geist Mono', 'JetBrains Mono', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },
      animation: {
        'pulse-dot': 'pulse-dot 2s infinite ease-in-out',
        'spin-slow': 'spin 30s linear infinite',
        'spin-slow-reverse': 'spin 20s linear infinite reverse',
      },
    },
  },
  plugins: [],
}
