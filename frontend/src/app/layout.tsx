import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'CARIS — Cedar Bayou Reliability Intelligence',
  description: 'CPChem Cedar Bayou Agentic Reliability Intelligence System by Frame Data AI',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
