<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
  <xsl:output method="html" indent="yes" />

  <xsl:template match="/">
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>
          <xsl:value-of select="claimWorkflow/@title" />
        </title>
        <style>
          body {
            margin: 0;
            font-family: "Segoe UI", sans-serif;
            color: #15262c;
            background: linear-gradient(180deg, #fbf7f1 0%, #f2ece3 100%);
          }
          main {
            width: min(860px, calc(100% - 2rem));
            margin: 0 auto;
            padding: 2rem 0 3rem;
          }
          .shell {
            background: rgba(255, 252, 247, 0.96);
            border: 1px solid rgba(20, 37, 44, 0.1);
            border-radius: 28px;
            box-shadow: 0 18px 48px rgba(20, 37, 44, 0.08);
            padding: 1.5rem;
          }
          .eyebrow {
            display: inline-flex;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            background: rgba(14, 108, 103, 0.1);
            color: #0e6c67;
            font-size: 0.74rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            font-weight: 700;
          }
          h1 {
            margin: 1rem 0 0.55rem;
            font-size: clamp(2rem, 4vw, 3rem);
            line-height: 1.05;
          }
          p {
            color: #5f6d73;
            line-height: 1.6;
          }
          ol {
            list-style: none;
            padding: 0;
            margin: 1.3rem 0 0;
            display: grid;
            gap: 0.9rem;
          }
          li {
            background: rgba(255, 255, 255, 0.54);
            border: 1px solid rgba(20, 37, 44, 0.08);
            border-radius: 20px;
            padding: 1rem;
          }
          .meta {
            font-size: 0.8rem;
            color: #0e6c67;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
          }
          h2 {
            margin: 0.4rem 0;
            font-size: 1.15rem;
          }
        </style>
      </head>
      <body>
        <main>
          <section class="shell">
            <span class="eyebrow">XML + DTD + XSL</span>
            <h1><xsl:value-of select="claimWorkflow/@title" /></h1>
            <p><xsl:value-of select="claimWorkflow/summary" /></p>
            <p>
              Owner:
              <strong><xsl:value-of select="claimWorkflow/@owner" /></strong>
            </p>
            <ol>
              <xsl:for-each select="claimWorkflow/step">
                <li>
                  <div class="meta">
                    Step <xsl:value-of select="@id" /> - <xsl:value-of select="@status" />
                  </div>
                  <h2><xsl:value-of select="title" /></h2>
                  <p><xsl:value-of select="detail" /></p>
                </li>
              </xsl:for-each>
            </ol>
          </section>
        </main>
      </body>
    </html>
  </xsl:template>
</xsl:stylesheet>
