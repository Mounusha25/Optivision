# OptiVision Full-Stack Deployment Guide

## 🚀 Complete Deployment with Backend Functionality

### **Architecture Overview**
- **Frontend**: Deployed on Vercel (Free)
- **Backend**: Deployed on Railway ($5/month)
- **Auto-detection**: Frontend automatically detects environment and connects to appropriate backend

---

## **Step 1: Deploy Backend to Railway**

### 1.1 Sign up for Railway
1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub account
3. Connect your GitHub repository

### 1.2 Deploy Backend
1. Create new project in Railway
2. Connect your GitHub repository: `Brijesh03032001/OptiVision`
3. Railway will automatically detect the `Dockerfile.railway` and deploy
4. Your backend will be available at: `https://optivision-backend.railway.app`

### 1.3 Configure Environment Variables (Optional)
```bash
MODEL_NAME=ggml-org/SmolVLM-500M-Instruct-GGUF
PORT=8080
```

---

## **Step 2: Deploy Frontend to Vercel**

### 2.1 Install Vercel CLI
```bash
npm install -g vercel
```

### 2.2 Deploy to Vercel
```bash
cd /Users/brijeshkumar03/Downloads/AIProject/OptiVision
vercel --prod
```

### 2.3 Configure Vercel Settings
1. Set **Root Directory**: `/` (default)
2. Set **Output Directory**: Leave empty (static files)
3. The frontend will automatically connect to Railway backend

---

## **Step 3: Alternative One-Click Deployments**

### **Option A: Vercel (Frontend) + Railway (Backend)**
[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/Brijesh03032001/OptiVision)

### **Option B: Netlify (Frontend Only)**
[![Deploy to Netlify](https://www.netlify.com/img/deploy/button.svg)](https://app.netlify.com/start/deploy?repository=https://github.com/Brijesh03032001/OptiVision)

---

## **Step 4: Manual Vercel Deployment**

### 4.1 Create Vercel Account
1. Go to [vercel.com](https://vercel.com)
2. Sign up with GitHub
3. Import your repository

### 4.2 Configure Build Settings
- **Framework Preset**: Other
- **Root Directory**: `/`
- **Build Command**: (leave empty)
- **Output Directory**: (leave empty)
- **Install Command**: (leave empty)

### 4.3 Environment Variables (if needed)
```bash
NEXT_PUBLIC_API_URL=https://your-railway-backend.railway.app
```

---

## **How Auto-Detection Works**

The frontend automatically detects the environment:

### **Production (Vercel/Netlify)**
- API URL: `https://optivision-backend.railway.app`
- CORS enabled for production domains

### **Development (Local)**
- API URL: `http://localhost:8080`
- Uses local backend server

### **Network (Local WiFi)**
- API URL: `http://[your-ip]:8080`
- Accessible from other devices

---

## **Cost Breakdown**

### **Free Tier**
- **Vercel**: Frontend hosting (Free)
- **Railway**: $5/month for backend
- **Total**: $5/month

### **Alternative Free Options**
- **GitHub Pages**: Frontend only
- **Netlify**: Frontend only
- **Backend**: Run locally or on free VPS

---

## **Domain Configuration (Optional)**

### **Custom Domain for Frontend**
1. Buy domain (e.g., `optivision.com`)
2. Add to Vercel settings
3. Update DNS records

### **Custom Domain for Backend**
1. Use Railway's custom domain feature
2. Or use Cloudflare for free SSL

---

## **Monitoring & Analytics**

### **Vercel Analytics**
- Built-in analytics
- Performance monitoring
- Error tracking

### **Railway Monitoring**
- Resource usage
- Deployment logs
- Health checks

---

## **Environment Variables Summary**

### **Frontend (Vercel)**
```bash
# Optional - will auto-detect if not set
NEXT_PUBLIC_API_URL=https://your-backend.railway.app
```

### **Backend (Railway)**
```bash
PORT=8080
MODEL_NAME=ggml-org/SmolVLM-500M-Instruct-GGUF
```

---

## **Troubleshooting**

### **Backend Not Starting**
- Check Railway logs
- Verify Docker build
- Check memory limits

### **Frontend Can't Connect**
- Verify CORS settings
- Check API URL in browser console
- Test backend endpoint directly

### **Performance Issues**
- Upgrade Railway plan
- Optimize model size
- Use CDN for frontend assets

---

## **Next Steps**

1. **Deploy Backend**: Follow Railway deployment
2. **Deploy Frontend**: Use Vercel one-click deploy
3. **Test Functionality**: Verify all features work
4. **Custom Domain**: Add your domain (optional)
5. **Monitoring**: Set up alerts and monitoring

Your OptiVision app will be fully functional with:
- ✅ Real-time vision processing
- ✅ Multi-source input support
- ✅ AI-powered analysis
- ✅ Global accessibility
- ✅ Professional deployment
