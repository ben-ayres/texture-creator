This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.

## Latest Patina test

This version includes two photo workflows:

- **Clean source**: the recommended default for a clear, nearly square-on photograph.
- **Difficult photo**: for angled or obstructed photographs; results need review.

It also applies the photo's EXIF orientation before processing, which prevents sideways camera images from being mistaken for a perspective-correction problem.

### Test steps

1. Deploy this package using the same Vercel/Render process as before.
2. Open the Patina page.
3. Choose **Stone pavers**.
4. Leave **Clean source** selected.
5. Upload the flooring photograph.
6. Click **Analyse source**.
7. Click **Auto-detect & flatten** only if the photograph needs a front-on preview.
8. Review the preview, confirm the crop, then create the texture.

Please check three things: whether the image is still sideways, whether the plant/furniture are included, and whether the pavers remain straight rather than visibly warped.

If the result is not suitable, do not repeatedly redeploy. Save a screenshot or download the result and send it back so the next change can be tested locally first.
