// Environment configuration for OptiVision
const config = {
  // Auto-detect environment and set API URL
  getApiUrl: () => {
    // Check if we're in production (Vercel)
    if (window.location.hostname.includes('vercel.app') || 
        window.location.hostname.includes('netlify.app') ||
        window.location.hostname !== 'localhost' && 
        window.location.hostname !== '127.0.0.1' &&
        !window.location.hostname.includes('192.168') &&
        !window.location.hostname.includes('10.')) {
      
      // Production: Use Railway backend
      return 'https://optivision-backend.railway.app';
    } 
    
    // Development: Use local backend
    return 'http://localhost:8080';
  },
  
  // API endpoints
  endpoints: {
    chat: '/v1/chat/completions',
    health: '/health'
  },
  
  // Check if backend is available
  async checkBackend() {
    try {
      const response = await fetch(this.getApiUrl() + this.endpoints.health, {
        method: 'GET',
        mode: 'cors'
      });
      return response.ok;
    } catch (error) {
      console.warn('Backend not available:', error);
      return false;
    }
  },
  
  // Get full API URL
  getEndpoint(endpoint) {
    return this.getApiUrl() + this.endpoints[endpoint];
  }
};

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
  module.exports = config;
} else {
  window.OptiVisionConfig = config;
}
