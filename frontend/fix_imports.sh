#!/bin/bash
# Fix type imports for files

# Badge
sed -i "1s/.*/import { forwardRef } from 'react';\nimport type { HTMLAttributes } from 'react';/" src/components/ui/Badge.tsx

# Spinner
sed -i "1s/.*/import { forwardRef } from 'react';\nimport type { HTMLAttributes } from 'react';/" src/components/ui/Spinner.tsx

# Progress
sed -i "1s/.*/import { forwardRef } from 'react';\nimport type { HTMLAttributes } from 'react';/" src/components/ui/Progress.tsx

# Modal
sed -i '1s/.*/import { useEffect } from '\''react'\'';\nimport type { ReactNode } from '\''react'\'';/' src/components/ui/Modal.tsx

# Toast
sed -i '1s/.*/import { useEffect } from '\''react'\'';\nimport type { ReactNode } from '\''react'\'';/' src/components/ui/Toast.tsx

# MainLayout
sed -i '1s/.*/import { useState } from '\''react'\'';\nimport type { ReactNode } from '\''react'\'';/' src/components/layout/MainLayout.tsx

# FileUpload - fix Button and FileRejection import
sed -i '2s/.*/import type { FileRejection } from '\''react-dropzone'\'';/' src/components/ui/FileUpload.tsx
sed -i '5d' src/components/ui/FileUpload.tsx  # Remove unused Button import

# Header - remove unused cn import
sed -i '/^import.*cn.*from/d' src/components/layout/Header.tsx

# Modal - remove unused Button import  
sed -i '/^import.*Button.*from/d' src/components/ui/Modal.tsx

# Upload - remove unused Badge import
sed -i '/^import.*Badge.*from.*Badge/d' src/pages/Upload.tsx

