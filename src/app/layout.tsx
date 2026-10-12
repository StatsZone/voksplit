import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Voksplit — by Sound Capsule | Séparation Vocale IA Locale",
  description: "Isolez les voix et instruments de vos chansons en temps réel, 100 % gratuit, sans cloud et respectueux de votre vie privée.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html
      lang="fr"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased dark`}
    >
      <head>
        <link rel="manifest" href="/manifest.json" />
        <meta name="theme-color" content="#06b6d4" />
      </head>
      <body className="min-h-full flex flex-col bg-slate-950 text-slate-100 selection:bg-cyan-500 selection:text-black">
        {children}
        <script
          dangerouslySetInnerHTML={{
            __html: `
              if ('serviceWorker' in navigator) {
                window.addEventListener('load', function() {
                  navigator.serviceWorker.register('/sw.js').catch(function(e) {
                    console.warn('[PWA] SW registration failed:', e);
                  });
                });
              }
            `,
          }}
        />
      </body>
    </html>
  );
}
