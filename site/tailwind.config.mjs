/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}'],
  theme: {
    extend: {
      colors: {
        ink: '#1d1d1f',
        'ink-secondary': '#6e6e73',
        paper: '#ffffff',
        'paper-alt': '#f5f5f7',
        accent: '#0066cc',
        rule: '#d2d2d7',
      },
      fontFamily: {
        sans: ['Inter Tight', 'Inter', 'system-ui', '-apple-system', 'sans-serif'],
      },
      maxWidth: {
        content: '980px',
      },
      letterSpacing: {
        tighter: '-0.02em',
        tightest: '-0.03em',
      },
    },
  },
  plugins: [],
};
