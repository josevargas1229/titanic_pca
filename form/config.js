class ConfigLoader {
    constructor() {
        this.config = {};
        this.loaded = false;
    }

    async loadConfig() {
        try {
            // Attempt to load from .env
            const response = await fetch('.env');
            if (response.ok) {
                const envContent = await response.text();
                this.parseEnvContent(envContent);
                console.log('✅ Configuration loaded from .env');
            } else {
                throw new Error('Could not load .env');
            }
        } catch (error) {
            console.warn('⚠️ Could not load .env, using default configuration');
            this.setDefaults();
        }
        
        this.loaded = true;
        return this.config;
    }

    parseEnvContent(content) {
        const lines = content.split('\n');
        lines.forEach(line => {
            line = line.trim();
            // Ignore comments and empty lines
            if (line && !line.startsWith('#')) {
                const [key, ...valueParts] = line.split('=');
                if (key && valueParts.length > 0) {
                    const value = valueParts.join('=').trim();
                    // Remove quotes if present
                    const cleanValue = value.replace(/^["']|["']$/g, '');
                    this.config[key.trim()] = cleanValue;
                }
            }
        });
    }

    setDefaults() {
        this.config = {
            API_URL: 'http://localhost:5000', // Update with your Flask server URL
            DEBUG: 'false',
        };
    }

    get(key, defaultValue = null) {
        return this.config[key] || defaultValue;
    }

    getApiUrl() {
        return this.get('API_URL', 'http://localhost:5000');
    }

    isDebug() {
        return this.get('DEBUG', 'false').toLowerCase() === 'true';
    }
}

// Create global instance
window.configLoader = new ConfigLoader();