// script.js — Smart Mirror dashboard: clock, status, weather, and schedule.

const timeEl = document.getElementById('time');
const dateEl = document.getElementById('date');
const greetingEl = document.getElementById('greeting');
const userNameEl = document.getElementById('user-name');
const esp32DotEl = document.getElementById('esp32-dot');
const esp32StatusEl = document.getElementById('esp32-status');
const cameraDotEl = document.getElementById('camera-dot');
const cameraStatusEl = document.getElementById('camera-status');
const privacyBannerEl = document.getElementById('privacy-banner');

// Weather elements
const weatherTempEl = document.getElementById('weather-temp');
const weatherIconEl = document.getElementById('weather-icon');
const weatherConditionEl = document.getElementById('weather-condition');
const weatherRangeEl = document.getElementById('weather-range');
const weatherCityEl = document.getElementById('weather-city');

// Calendar elements
const calendarEventsEl = document.getElementById('calendar-events');

let currentUserId = null;

const WEATHER_ICONS = {
  'clear': { day: '☀️', night: '🌙' },
  'partly-cloudy': { day: '⛅', night: '☁️' },
  'cloudy': { day: '☁️', night: '☁️' },
  'fog': { day: '🌫️', night: '🌫️' },
  'drizzle': { day: '🌦️', night: '🌧️' },
  'rain': { day: '🌧️', night: '🌧️' },
  'showers': { day: '🌧️', night: '🌧️' },
  'thunderstorm': { day: '⛈️', night: '⛈️' },
  'snow': { day: '🌨️', night: '🌨️' },
};

function updateClock() {
  const now = new Date();
  const hours = String(now.getHours()).padStart(2, '0');
  const minutes = String(now.getMinutes()).padStart(2, '0');
  timeEl.textContent = `${hours}:${minutes}`;

  dateEl.textContent = now.toLocaleDateString(undefined, {
    weekday: 'long',
    month: 'long',
    day: 'numeric',
  });
}

function updateGreeting() {
  const hour = new Date().getHours();
  let text;
  if (hour < 5) text = 'Still up?';
  else if (hour < 12) text = 'Good morning.';
  else if (hour < 17) text = 'Good afternoon.';
  else if (hour < 21) text = 'Good evening.';
  else text = 'Good night.';
  greetingEl.textContent = text;
}

async function updateWeather() {
  if (!weatherTempEl) return;
  try {
    const res = await fetch('/api/weather');
    const data = await res.json();

    weatherTempEl.textContent = `${data.temperature}°`;
    weatherConditionEl.textContent = data.condition || 'Clear';
    weatherCityEl.textContent = data.city || '';

    if (data.high !== null && data.low !== null) {
      weatherRangeEl.textContent = `H: ${data.high}° L: ${data.low}°`;
    } else {
      weatherRangeEl.textContent = `${data.humidity}% Humidity`;
    }

    const iconEntry = WEATHER_ICONS[data.icon] || { day: '☀️', night: '🌙' };
    weatherIconEl.textContent = data.is_day ? iconEntry.day : iconEntry.night;
  } catch (err) {
    weatherConditionEl.textContent = 'Weather offline';
  }
}

async function updateCalendar(userId) {
  if (!calendarEventsEl) return;
  try {
    const targetUser = userId || currentUserId || '';
    const url = targetUser ? `/api/calendar?user_id=${encodeURIComponent(targetUser)}` : '/api/calendar';
    const res = await fetch(url);
    const data = await res.json();

    const events = data.events || [];
    if (events.length === 0) {
      calendarEventsEl.innerHTML = '<div class="calendar-empty">No events scheduled today</div>';
      return;
    }

    calendarEventsEl.innerHTML = events.map(evt => `
      <div class="calendar-item">
        <span class="calendar-time">${escapeHtml(evt.time)}</span>
        <span class="calendar-title">${escapeHtml(evt.title)}</span>
      </div>
    `).join('');
  } catch (err) {
    calendarEventsEl.innerHTML = '<div class="calendar-empty">Schedule unavailable</div>';
  }
}

function escapeHtml(str) {
  if (!str) return '';
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

async function updateStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();

    const newUserId = data.user.id || (data.user.name || '').toLowerCase().replace(/\s+/g, '_');
    userNameEl.textContent = data.user.name;

    // If active user switched, refresh calendar immediately
    if (newUserId && newUserId !== currentUserId) {
      currentUserId = newUserId;
      updateCalendar(currentUserId);
      updateOutfit(currentUserId);
    }

    // Privacy mode takes visual precedence — banner shows whenever active.
    privacyBannerEl.style.display = data.privacy_mode ? 'block' : 'none';

    // Voice concierge & assistant overlay
    const voiceOverlayEl = document.getElementById('voice-overlay');
    const voiceDotEl = document.getElementById('voice-dot');
    const voiceStateTextEl = document.getElementById('voice-state-text');
    const voiceTranscriptEl = document.getElementById('voice-transcript');
    const voiceReplyEl = document.getElementById('voice-reply');
    const soundwaveBarsEl = document.getElementById('soundwave-bars');

    if (data.voice && data.voice.state && data.voice.state !== 'idle') {
      voiceOverlayEl.style.display = 'block';
      if (voiceDotEl) voiceDotEl.className = 'voice-dot ' + data.voice.state;
      if (soundwaveBarsEl) {
        if (data.voice.state === 'speaking') {
          soundwaveBarsEl.classList.add('active');
        } else {
          soundwaveBarsEl.classList.remove('active');
        }
      }
      if (voiceStateTextEl) {
        voiceStateTextEl.textContent = data.voice.state === 'speaking' ? 'Speaking Briefing...' : (data.voice.state + '...');
      }
      if (voiceTranscriptEl) {
        if (data.voice.text && data.voice.text !== 'Voice Concierge Briefing') {
          voiceTranscriptEl.style.display = 'block';
          voiceTranscriptEl.textContent = `"${data.voice.text}"`;
        } else {
          voiceTranscriptEl.style.display = 'none';
        }
      }
      if (voiceReplyEl) {
        voiceReplyEl.textContent = data.voice.reply || '';
      }
    } else {
      voiceOverlayEl.style.display = 'none';
      if (soundwaveBarsEl) soundwaveBarsEl.classList.remove('active');
    }

    // Live Camera Clothing & AI Stylist pill
    const stylistPillEl = document.getElementById('stylist-live-pill');
    if (stylistPillEl) {
      if (data.stylist && data.stylist.detected && data.stylist.detected.name) {
        stylistPillEl.style.display = 'inline-block';
        stylistPillEl.textContent = `👁️ Wearing: ${data.stylist.detected.name}`;
      } else {
        stylistPillEl.style.display = 'none';
      }
    }

    // Live Camera Grooming & Wellness pill
    const groomingPillEl = document.getElementById('grooming-pill');
    if (groomingPillEl) {
      if (data.grooming && data.grooming.status === 'ok') {
        const g = data.grooming;
        const parts = [];
        if (g.well_groomed) parts.push('✨ Well-Groomed');
        if (g.facial_hair && g.facial_hair !== 'Clean shaven') parts.push(g.facial_hair);
        if (g.skin && g.skin.includes('Clear')) parts.push('Clear Skin');
        if (g.accessories && g.accessories.includes('Glasses')) parts.push('👓');

        const label = parts.length ? parts.slice(0, 2).join(' · ') : (g.facial_hair || 'Looking Fresh');
        groomingPillEl.style.display = 'inline-block';
        groomingPillEl.textContent = label;
      } else {
        groomingPillEl.style.display = 'none';
      }
    }

    // ESP32 telemetry: live room temperature, humidity, light, motion
    const indoorClimateEl = document.getElementById('indoor-climate');
    const indoorTempEl = document.getElementById('indoor-temp');
    const indoorHumEl = document.getElementById('indoor-hum');
    const sensorReadingsEl = document.getElementById('sensor-readings');
    const sensorTempHumEl = document.getElementById('sensor-temp-hum');
    const sensorLightMotionEl = document.getElementById('sensor-light-motion');

    if (data.sensors && data.sensors.temperature !== null && data.sensors.temperature !== undefined) {
      const tempVal = Number(data.sensors.temperature).toFixed(1);
      const humVal = (data.sensors.humidity !== null && data.sensors.humidity !== undefined)
        ? Math.round(Number(data.sensors.humidity))
        : null;

      if (indoorClimateEl && indoorTempEl) {
        indoorClimateEl.style.display = 'flex';
        indoorTempEl.textContent = `${tempVal}°C`;
        if (indoorHumEl && humVal !== null) {
          indoorHumEl.textContent = `${humVal}% Hum`;
        }
      }

      if (sensorReadingsEl && sensorTempHumEl) {
        sensorReadingsEl.style.display = 'flex';
        sensorTempHumEl.textContent = humVal !== null ? `${tempVal}°C · ${humVal}% Hum` : `${tempVal}°C`;
        if (sensorLightMotionEl) {
          const lightStr = data.sensors.light ? data.sensors.light.toUpperCase() : 'LIGHT';
          const motionStr = data.sensors.motion ? 'MOTION' : 'STILL';
          sensorLightMotionEl.textContent = `${lightStr} · ${motionStr}`;
        }
      }

      esp32StatusEl.textContent = `ESP32: ${data.system.esp32_status} (${tempVal}°C)`;
    } else {
      if (indoorClimateEl) indoorClimateEl.style.display = 'none';
      if (sensorReadingsEl) sensorReadingsEl.style.display = 'none';
      esp32StatusEl.textContent = `ESP32: ${data.system.esp32_status}`;
    }

    esp32DotEl.className = 'status-dot ' + (
      data.system.esp32_status === 'online' ? 'connected' : (data.system.esp32_mock_mode ? 'mock' : '')
    );

    cameraStatusEl.textContent = `Camera: ${data.system.camera_status}`;
    cameraDotEl.className = 'status-dot ' + (
      data.system.camera_status === 'online' ? 'connected' : (data.system.camera_mock_mode ? 'mock' : '')
    );
  } catch (err) {
    esp32StatusEl.textContent = 'ESP32: unavailable';
    cameraStatusEl.textContent = 'Camera: unavailable';
  }
}

async function updateOutfit(userId) {
  const outfitItemsEl = document.getElementById('outfit-items');
  const outfitReasonEl = document.getElementById('outfit-reason');
  const stylistPillEl = document.getElementById('stylist-live-pill');
  if (!outfitItemsEl) return;

  try {
    const targetUser = userId || currentUserId || '';
    const url = targetUser ? `/api/outfit?user_id=${encodeURIComponent(targetUser)}` : '/api/outfit';
    const res = await fetch(url);
    const data = await res.json();

    if (data.status === 'empty_wardrobe') {
      outfitItemsEl.innerHTML = `<div class="outfit-empty">Wardrobe empty.<br>Go to <strong>/wardrobe</strong> on your phone to add clothes.</div>`;
      outfitReasonEl.textContent = '';
      return;
    }

    if (!data.items || data.items.length === 0) {
      outfitItemsEl.innerHTML = '<div class="outfit-empty">No suitable outfit found.</div>';
      outfitReasonEl.textContent = '';
      return;
    }

    const CATEGORY_EMOJI = {
      top: '👕', bottom: '👖', footwear: '👟', outerwear: '🧥', accessory: '💍'
    };

    outfitItemsEl.innerHTML = data.items.map(item => {
      const imgHtml = item.image
        ? `<img src="/wardrobe-images/${item.user_id}/${item.image}" class="outfit-card-img" />`
        : `<div class="outfit-card-emoji">${CATEGORY_EMOJI[item.category] || '👔'}</div>`;
      
      return `
        <div class="outfit-card">
          ${imgHtml}
          <div class="outfit-card-info">
            <div class="outfit-card-label">${item.label}</div>
            <div class="outfit-card-name">${escapeHtml(item.name)}</div>
          </div>
        </div>
      `;
    }).join('');

    outfitReasonEl.textContent = data.reason || '';

    // Fetch live AI stylist advice based on camera detection
    try {
      const sUrl = targetUser ? `/api/mirror/clothing/status?user_id=${encodeURIComponent(targetUser)}` : '/api/mirror/clothing/status';
      const sRes = await fetch(sUrl);
      const sData = await sRes.json();
      if (sData && sData.advice) {
        outfitReasonEl.innerHTML = `<span style="color: var(--ember);">💡 Stylist:</span> ${escapeHtml(sData.advice)}`;
        if (stylistPillEl && sData.detected && sData.detected.name) {
          stylistPillEl.style.display = 'inline-block';
          stylistPillEl.textContent = `👁️ Wearing: ${sData.detected.name}`;
        }
      }
    } catch (sErr) {
      // Fallback to standard reason
    }
  } catch (err) {
    outfitItemsEl.innerHTML = '<div class="outfit-empty">Outfit engine offline</div>';
  }
}

// Initial load
updateClock();
updateGreeting();
updateStatus();
updateWeather();
updateCalendar();
updateOutfit();

// Timers
setInterval(updateClock, 1000 * 30);
setInterval(updateGreeting, 1000 * 60 * 15);
setInterval(updateStatus, 1000 * 5);       // Poll active user and privacy status every 5s
setInterval(updateCalendar, 1000 * 30);     // Periodic calendar refresh
setInterval(updateOutfit, 1000 * 60 * 30);  // Outfit refresh every 30 mins
setInterval(updateWeather, 1000 * 60 * 10); // Weather refresh every 10 mins

