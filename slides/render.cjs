// npm install --no-save playwright, then node slides/render.cjs from repository root.
// Requires Google Chrome. Writes the nine-slide PDF and checks text fits each slide.
const { chromium } = require('playwright');
const path = require('node:path');
const fs = require('node:fs');
(async () => {
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
    await page.goto('file://' + path.join(__dirname, 'index.html'));
    await page.evaluate(() => document.fonts.ready);
    if (!(await page.evaluate(() => document.fonts.check('16px Inter')))) throw new Error('Bundled Inter font did not load');
    const count = await page.locator('.slide').count();
    if (count !== 9) throw new Error(`Expected nine slides, found ${count}`);
    const failures = [];
    for (let i = 0; i < count; i++) {
      await page.evaluate(n => show(n), i);
      const overflow = await page.locator('.slide.active').evaluate(slide => {
        const s = slide.getBoundingClientRect();
        // Text must stay inside the slide and inside its own box.
        const clipped = [...slide.querySelectorAll('h1,h2,p,li,td,th,dt,dd,figcaption,.hero,.foot,.equation,svg text')].filter(el => {
          const r = el.getBoundingClientRect();
          return r.left < s.left - 1 || r.right > s.right + 1 || r.bottom > s.bottom + 1 || el.scrollWidth > el.clientWidth + 2;
        }).map(el => el.textContent);
        // Content must not run into the footer.
        const body = slide.querySelector('.body');
        const limit = body.getBoundingClientRect().bottom;
        const spill = [...body.querySelectorAll('*')].filter(el => el.getBoundingClientRect().bottom > limit + 1).map(el => `Runs into footer: ${el.textContent.trim().slice(0, 60)}`);
        return [...clipped, ...spill];
      });
      if (overflow.length) failures.push({ slide: i + 1, overflow });
      if (process.env.SLIDE_PREVIEWS) {
        fs.mkdirSync(process.env.SLIDE_PREVIEWS, { recursive: true });
        await page.screenshot({ path: path.join(process.env.SLIDE_PREVIEWS, `${i + 1}.png`) });
      }
    }
    if (failures.length) throw new Error(JSON.stringify(failures, null, 1));
    await page.pdf({ path: path.join(__dirname, 'Unblo.pdf'), preferCSSPageSize: true, printBackground: true });
    console.log('PASS: nine slides rendered, three per task; no text overflow. PDF: slides/Unblo.pdf');
  } finally { await browser.close(); }
})().catch(e => { console.error(e); process.exitCode = 1; });
