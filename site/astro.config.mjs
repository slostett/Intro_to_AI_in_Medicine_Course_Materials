import { defineConfig } from 'astro/config';
import tailwind from '@astrojs/tailwind';

export default defineConfig({
  site: 'https://slostett.github.io/Intro_to_AI_in_Medicine_Course_Materials',
  base: '/Intro_to_AI_in_Medicine_Course_Materials',
  integrations: [tailwind()],
});
