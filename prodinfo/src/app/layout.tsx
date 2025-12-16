import type { Metadata } from "next";
import "./globals.css";
import AppShell from "@/components/app-shell";

export const metadata: Metadata = {
  title: "IngrediScore Web",
  description: "Yuka-like ingredient scanner with iOS polish.",
  applicationName: "IngrediScore Web",
  manifest: "/manifest.webmanifest",
  appleWebApp: {
    capable: true,
    statusBarStyle: "black-translucent",
    title: "IngrediScore Web",
  },
  themeColor: "#0b1021",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <meta name="apple-mobile-web-app-capable" content="yes" />
        <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent" />
        <meta name="apple-mobile-web-app-title" content="IngrediScore Web" />
      </head>
      <body className="antialiased ios-safe-area">
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
