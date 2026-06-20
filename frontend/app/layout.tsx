import './globals.css'
import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'Pro Swing AI — Kinetic Elite',
  description: 'Elite athletic coaching powered by advanced kinematic analysis, real-time AI computer vision, and biomechanical intelligence.',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
      </head>
      <body className="custom-scrollbar">{children}</body>
    </html>
  )
}
