"""
app/assistant/brain.py — Centralized Voice Brain & Database Knowledge Hub for ReflectAI.

Connects the voice assistant to *everything* across ReflectAI:
- User profile & preferences (users, preferences, profiles.json)
- Reminders (reminders table — add, list, delete, complete)
- Wardrobe & Outfits (wardrobe, outfit_log, OutfitEngine — counts, wear stats, recommendations)
- ESP32 Telemetry (temperature, humidity, motion, light, connection status)
- Live Camera Vision (MirrorStylist detected clothing, GroomingClassifier hair/skin/beard analysis)
- Live Weather (WeatherService outdoor conditions, high/low, rain forecast)
- Calendar & Schedule (CalendarService events today)
- System Settings & Privacy (SettingsRepository privacy_mode toggle, active user switching, user list)

Uses a Hybrid Architecture:
1. Fast-Path Action & Exact Query Engine (deterministic, 0ms LLM latency, zero hallucination)
2. Dynamic Knowledge Grounding + Local LLM (for open-ended conversational reasoning)
"""

import json
import logging
import os
import re
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional, Tuple

from app.calendar import CalendarService
from app.config.settings import settings
from app.database.repositories import (
    OutfitLogRepository,
    ReminderRepository,
    SettingsRepository,
    UserRepository,
    WardrobeRepository,
)
from app.recommendations import OutfitEngine
from app.weather import WeatherService

logger = logging.getLogger("reflectai.voice_brain")

# LLM singleton if llama-cpp is loaded locally
_local_llm = None
_local_llm_attempted = False


def _get_local_llm():
    global _local_llm, _local_llm_attempted
    if _local_llm is not None:
        return _local_llm
    if _local_llm_attempted:
        return None

    _local_llm_attempted = True
    try:
        from llama_cpp import Llama

        candidates = [
            settings.PROJECT_ROOT / "voice_engine" / "qwen2.5-3b-instruct-q4_k_m.gguf",
            settings.PROJECT_ROOT / "voice_engine" / "qwen2.5-1.5b-instruct-q4_k_m.gguf",
            settings.PROJECT_ROOT / "voice_engine" / "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        ]
        chosen = None
        for c in candidates:
            if c.exists() and c.stat().st_size > 100 * 1024 * 1024:
                chosen = c
                break

        if chosen:
            logger.info(f"Loading local LLM for VoiceBrain: {chosen.name}")
            _local_llm = Llama(
                model_path=str(chosen),
                n_ctx=2048,
                n_threads=4,
                verbose=False,
            )
            return _local_llm
    except Exception as e:
        logger.warning(f"Could not initialize local LLM in VoiceBrain: {e}")
    return None


class VoiceBrain:
    @classmethod
    def get_knowledge_snapshot(
        cls,
        user_id: Optional[str] = None,
        esp32_data: Optional[Dict[str, Any]] = None,
        stylist_data: Optional[Dict[str, Any]] = None,
        grooming_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Gathers a complete, live snapshot of everything across ReflectAI:
        User, registered users, reminders, wardrobe, outfit log, ESP32, vision, weather, calendar, settings.
        """
        if not user_id:
            user_id = SettingsRepository.get("active_user", default="guest")
        user_id = (user_id or "guest").strip()

        # 1. User Profile & Registered Users
        user = UserRepository.get(user_id)
        user_name = user["name"] if user else user_id.title()
        all_users = UserRepository.list_all()

        profiles_file = settings.DATA_DIR / "profiles.json"
        profile_json = {}
        if profiles_file.exists():
            try:
                with open(profiles_file, "r", encoding="utf-8") as f:
                    profile_json = json.load(f).get(user_id.lower(), {})
            except Exception:
                pass

        user_info = {
            "id": user_id,
            "name": user_name,
            "preferred_style": (user.get("preferred_style") if user else None) or "Casual",
            "hair_preference": (user.get("hair_preference") if user else None) or profile_json.get("hair", ""),
            "temperature_unit": (user.get("temperature_unit") if user else None) or "C",
            "created_at": user.get("created_at") if user else None,
            "favorite_pieces": profile_json.get("outfit", []),
        }

        # 2. Reminders
        reminders_active = ReminderRepository.list_for_user(user_id, include_completed=False)
        reminders_all = ReminderRepository.list_for_user(user_id, include_completed=True)
        reminders_completed = [r for r in reminders_all if r.get("completed")]

        # 3. Wardrobe Inventory
        wardrobe_items = WardrobeRepository.list_for_user(user_id, active_only=True)
        categories = {}
        colors = set()
        most_worn_item = None
        max_worn = -1

        for item in wardrobe_items:
            cat = (item.get("category") or "other").lower()
            categories[cat] = categories.get(cat, 0) + 1
            if item.get("color"):
                colors.add(item["color"].title())
            times = item.get("times_worn") or 0
            if times > max_worn:
                max_worn = times
                most_worn_item = item

        # 4. Outfits & History
        today_recommendation = {}
        try:
            today_recommendation = OutfitEngine.recommend(user_id)
        except Exception as e:
            logger.warning(f"Error computing outfit recommendation: {e}")

        recent_outfit_logs = []
        try:
            recent_outfit_logs = OutfitLogRepository.get_recent(user_id, days=7)
        except Exception:
            pass

        # 5. Outdoor Weather
        weather = {}
        try:
            weather = WeatherService.get_weather()
        except Exception:
            pass

        # 6. ESP32 Room Climate
        esp32 = esp32_data or {}

        # 7. Calendar
        calendar_events = []
        try:
            calendar_events = CalendarService.get_events_for_user(user_id)
        except Exception:
            pass

        # 8. Live Vision
        vision = {
            "stylist": stylist_data or {},
            "grooming": grooming_data or {},
        }

        # 9. System Settings
        privacy_mode = SettingsRepository.get("privacy_mode", default="false") == "true"
        active_user_setting = SettingsRepository.get("active_user", default="guest")

        return {
            "active_user": user_info,
            "registered_users": [u["name"] for u in all_users],
            "reminders": {
                "active": reminders_active,
                "completed": reminders_completed,
            },
            "wardrobe": {
                "total_items": len(wardrobe_items),
                "items": wardrobe_items,
                "category_counts": categories,
                "colors": sorted(list(colors)),
                "most_worn": most_worn_item,
            },
            "outfit": {
                "recommendation": today_recommendation,
                "recent_history": recent_outfit_logs,
            },
            "esp32": esp32,
            "weather": weather,
            "calendar": calendar_events,
            "vision": vision,
            "system": {
                "privacy_mode": privacy_mode,
                "active_user_id": active_user_setting,
            },
            "timestamp": datetime.now().isoformat(),
        }

    @classmethod
    def execute_action_or_query(
        cls,
        query: str,
        user_id: Optional[str] = None,
        esp32_data: Optional[Dict[str, Any]] = None,
        stylist_data: Optional[Dict[str, Any]] = None,
        grooming_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Fast-Path Intent & Direct Database Action Router.
        Returns a dict {"reply": str, "action": str, "data": Any} if handled directly,
        or None if it should fall back to grounded LLM reasoning.
        """
        if not user_id:
            user_id = SettingsRepository.get("active_user", default="guest")
        q = query.strip().lower()
        # Clean punctuation for easier matching
        clean_q = re.sub(r"[^\w\s]", " ", q)
        clean_q = re.sub(r"\s+", " ", clean_q).strip()

        # ======================================================================
        # 1. REMINDERS ACTIONS & QUERIES
        # ======================================================================

        # 1A. Add Reminder
        # e.g.: "add reminder to buy groceries at 5 pm", "remind me to call mom at 3pm", "add reminder project work"
        m_add = (
            re.search(r"(?:add\s+reminder|remind\s+me)\s+(?:to\s+)?(.+?)\s+(?:at|for|by)\s+([0-9]{1,2}(?::[0-9]{2})?\s*(?:am|pm)?)", clean_q)
            or re.search(r"(?:add\s+reminder|remind\s+me)\s+(?:to\s+)?(.+)", clean_q)
        )
        if m_add and ("add reminder" in clean_q or "remind me" in clean_q):
            title = m_add.group(1).strip()
            time_str = m_add.group(2).strip() if len(m_add.groups()) > 1 and m_add.group(2) else "Today"
            # Filter out non-title words
            title = re.sub(r"^(please\s+|can\s+you\s+)", "", title).strip()
            if title:
                ReminderRepository.create(user_id, title.capitalize(), time_str)
                return {
                    "reply": f"I've added a reminder for {title.capitalize()} at {time_str}.",
                    "action": "add_reminder",
                    "data": {"title": title, "datetime": time_str, "user_id": user_id},
                }

        # 1B. Delete / Complete Reminder
        # e.g.: "delete reminder project work", "complete reminder buy milk", "mark reminder project work as done"
        m_del = re.search(r"(?:delete|remove|clear|cancel)\s+reminder\s+(.+)", clean_q)
        if m_del:
            target = m_del.group(1).strip().lower()
            reminders = ReminderRepository.list_for_user(user_id, include_completed=True)
            matched = next((r for r in reminders if target in r["title"].lower()), None)
            if matched:
                ReminderRepository.delete(matched["id"])
                return {
                    "reply": f"I have deleted the reminder: {matched['title']}.",
                    "action": "delete_reminder",
                    "data": matched,
                }
            return {
                "reply": f"I couldn't find a reminder matching '{target}'.",
                "action": "delete_reminder_not_found",
            }

        m_comp = re.search(r"(?:complete|mark|finish)\s+reminder\s+(.+?)(?:\s+as\s+done)?$", clean_q)
        if m_comp:
            target = m_comp.group(1).strip().lower()
            reminders = ReminderRepository.list_for_user(user_id, include_completed=False)
            matched = next((r for r in reminders if target in r["title"].lower()), None)
            if matched:
                ReminderRepository.complete(matched["id"])
                return {
                    "reply": f"I've marked your reminder '{matched['title']}' as completed.",
                    "action": "complete_reminder",
                    "data": matched,
                }
            return {
                "reply": f"I couldn't find a pending reminder matching '{target}'.",
                "action": "complete_reminder_not_found",
            }

        # 1C. List Reminders
        if any(p in clean_q for p in ["what are my reminders", "list my reminders", "do i have any reminders", "my reminders", "show reminders", "what tasks do i have"]):
            reminders = ReminderRepository.list_for_user(user_id, include_completed=False)
            if not reminders:
                return {
                    "reply": "You have no pending reminders.",
                    "action": "list_reminders",
                    "data": [],
                }
            if len(reminders) == 1:
                r = reminders[0]
                return {
                    "reply": f"You have 1 reminder: {r['title']} at {r['datetime']}.",
                    "action": "list_reminders",
                    "data": reminders,
                }
            rem_list = ", ".join([f"{r['title']} at {r['datetime']}" for r in reminders])
            return {
                "reply": f"You have {len(reminders)} reminders: {rem_list}.",
                "action": "list_reminders",
                "data": reminders,
            }

        # ======================================================================
        # 2. PRIVACY MODE CONTROL
        # ======================================================================
        if any(p in clean_q for p in ["turn on privacy mode", "enable privacy mode", "enable privacy", "privacy on", "turn privacy on"]):
            SettingsRepository.set("privacy_mode", "true")
            return {
                "reply": "Privacy mode is now enabled. Camera feeds and personal details are hidden.",
                "action": "set_privacy_mode",
                "data": {"privacy_mode": True},
            }

        if any(p in clean_q for p in ["turn off privacy mode", "disable privacy mode", "disable privacy", "privacy off", "turn privacy off"]):
            SettingsRepository.set("privacy_mode", "false")
            return {
                "reply": "Privacy mode is now disabled. Full smart mirror features are active.",
                "action": "set_privacy_mode",
                "data": {"privacy_mode": False},
            }

        if any(p in clean_q for p in ["is privacy mode on", "is privacy on", "privacy status", "check privacy"]):
            is_priv = SettingsRepository.get("privacy_mode", default="false") == "true"
            status_text = "enabled" if is_priv else "disabled"
            return {
                "reply": f"Privacy mode is currently {status_text}.",
                "action": "get_privacy_mode",
                "data": {"privacy_mode": is_priv},
            }

        # ======================================================================
        # 3. USER PROFILE & SWITCHING
        # ======================================================================

        # 3A. Switch User
        m_switch = re.search(r"(?:switch\s+(?:user\s+)?to|log\s*in\s+as)\s+([a-zA-Z0-9_\s]+)", clean_q)
        if m_switch:
            target_name = m_switch.group(1).strip()
            users = UserRepository.list_all()
            matched_user = next(
                (u for u in users if u["name"].lower() == target_name.lower() or u["id"].lower() == target_name.lower()),
                None
            )
            if matched_user:
                SettingsRepository.set("active_user", matched_user["id"])
                return {
                    "reply": f"Switched active user to {matched_user['name']}.",
                    "action": "switch_user",
                    "data": matched_user,
                }
            else:
                user_names = ", ".join([u["name"] for u in users])
                return {
                    "reply": f"I couldn't find user '{target_name}'. Available registered users are: {user_names}.",
                    "action": "switch_user_failed",
                }

        # 3B. Who is registered / List Users
        if any(p in clean_q for p in ["who is registered", "list all users", "list users", "registered users", "what users are on this mirror", "who has an account"]):
            users = UserRepository.list_all()
            user_names = ", ".join([u["name"] for u in users])
            return {
                "reply": f"The registered users on ReflectAI are: {user_names}.",
                "action": "list_users",
                "data": users,
            }

        # 3C. Who am I / Active User
        if any(p in clean_q for p in ["who am i", "what is my name", "who is logged in", "who is the active user", "what is my profile"]):
            user = UserRepository.get(user_id)
            name = user["name"] if user else user_id.title()
            style = (user.get("preferred_style") if user else None) or "Casual"
            hair = (user.get("hair_preference") if user else None) or "Standard"
            return {
                "reply": f"You are {name}, currently logged in with a {style} style preference and {hair} hair preference.",
                "action": "get_user_profile",
                "data": user,
            }

        # ======================================================================
        # 4. WARDROBE & CLOTHES QUERIES
        # ======================================================================

        # 4A. Total Count or Category Counts
        # e.g.: "how many clothes do i have", "how many shirts do i have", "how many pants do i have"
        m_count = re.search(r"how\s+many\s+([a-zA-Z\s]+?)\s+do\s+i\s+have", clean_q)
        if m_count:
            target_cat = m_count.group(1).strip().lower()
            items = WardrobeRepository.list_for_user(user_id, active_only=True)
            
            if any(w in target_cat for w in ["clothes", "items", "outfits", "pieces", "things in my wardrobe"]):
                return {
                    "reply": f"You have {len(items)} items in your wardrobe.",
                    "action": "wardrobe_count",
                    "data": {"count": len(items)},
                }
            
            # Map common category words
            cat_filter = None
            if any(w in target_cat for w in ["shirt", "top", "tshirt", "t shirt", "hoodie", "sweater"]):
                cat_filter = "top"
            elif any(w in target_cat for w in ["pant", "jean", "bottom", "trousers", "track pant", "shorts"]):
                cat_filter = "bottom"
            elif any(w in target_cat for w in ["shoe", "footwear", "sneaker", "boot"]):
                cat_filter = "footwear"
            elif any(w in target_cat for w in ["jacket", "coat", "outerwear", "blazer"]):
                cat_filter = "outerwear"

            if cat_filter:
                filtered = [i for i in items if i.get("category", "").lower() == cat_filter]
                names = ", ".join([i["name"] for i in filtered]) if filtered else "none"
                return {
                    "reply": f"You have {len(filtered)} {cat_filter}s: {names}.",
                    "action": "wardrobe_category_count",
                    "data": {"category": cat_filter, "count": len(filtered), "items": filtered},
                }

        # 4B. What clothes do I have / List wardrobe
        if any(p in clean_q for p in ["what clothes do i have", "list my wardrobe", "what is in my wardrobe", "show my wardrobe", "tell me my clothes", "what clothes are in my wardrobe"]):
            items = WardrobeRepository.list_for_user(user_id, active_only=True)
            if not items:
                return {
                    "reply": "Your wardrobe is currently empty. You can add items in the Wardrobe Manager.",
                    "action": "list_wardrobe",
                    "data": [],
                }
            tops = [i["name"] for i in items if i.get("category", "").lower() == "top"]
            bottoms = [i["name"] for i in items if i.get("category", "").lower() == "bottom"]
            others = [i["name"] for i in items if i.get("category", "").lower() not in ["top", "bottom"]]
            
            parts = [f"You have {len(items)} items in your wardrobe."]
            if tops:
                parts.append(f"Tops: {', '.join(tops[:4])}.")
            if bottoms:
                parts.append(f"Bottoms: {', '.join(bottoms[:4])}.")
            if others:
                parts.append(f"Other pieces: {', '.join(others[:3])}.")
            return {
                "reply": " ".join(parts),
                "action": "list_wardrobe",
                "data": items,
            }

        # 4C. Wear Frequency for specific item
        # e.g.: "how many times have i worn my white tshirt", "how often have i worn black jeans"
        m_wear_count = re.search(r"how\s+(?:many\s+times|often)\s+(?:have\s+i\s+worn|did\s+i\s+wear)\s+(?:my\s+)?(.+)", clean_q)
        if m_wear_count:
            target_item = m_wear_count.group(1).strip().lower()
            items = WardrobeRepository.list_for_user(user_id, active_only=True)
            matched = next((i for i in items if target_item in i["name"].lower()), None)
            if matched:
                times = matched.get("times_worn") or 0
                last = matched.get("last_worn") or "never"
                return {
                    "reply": f"You've worn your {matched['name']} {times} times, most recently on {last}.",
                    "action": "item_wear_frequency",
                    "data": matched,
                }

        # 4D. Last worn date for specific item
        m_last_worn = re.search(r"when\s+(?:did\s+i\s+last\s+wear|was\s+the\s+last\s+time\s+i\s+wore)\s+(?:my\s+)?(.+)", clean_q)
        if m_last_worn:
            target_item = m_last_worn.group(1).strip().lower()
            items = WardrobeRepository.list_for_user(user_id, active_only=True)
            matched = next((i for i in items if target_item in i["name"].lower()), None)
            if matched:
                last = matched.get("last_worn")
                if last:
                    return {
                        "reply": f"You last wore your {matched['name']} on {last}.",
                        "action": "item_last_worn",
                        "data": matched,
                    }
                else:
                    return {
                        "reply": f"You haven't logged wearing your {matched['name']} yet.",
                        "action": "item_last_worn",
                        "data": matched,
                    }

        # 4E. Most worn item
        if any(p in clean_q for p in ["most worn", "favorite clothes", "what do i wear the most", "clothes i wear most"]):
            items = WardrobeRepository.list_for_user(user_id, active_only=True)
            if items:
                sorted_items = sorted(items, key=lambda x: x.get("times_worn") or 0, reverse=True)
                top_item = sorted_items[0]
                times = top_item.get("times_worn") or 0
                return {
                    "reply": f"Your most worn item is your {top_item['name']}, which you have worn {times} times.",
                    "action": "most_worn_item",
                    "data": top_item,
                }

        # 4F. What did I wear yesterday / recently
        if any(p in clean_q for p in ["what did i wear yesterday", "what did i wear recently", "outfit history", "what did i wear"]):
            recent_logs = OutfitLogRepository.get_recent(user_id, days=7)
            if recent_logs:
                items = {i["id"]: i["name"] for i in WardrobeRepository.list_for_user(user_id, active_only=False)}
                log = recent_logs[0]
                pieces = []
                for k in ["top_id", "bottom_id", "footwear_id", "outerwear_id"]:
                    if log.get(k) and log[k] in items:
                        pieces.append(items[log[k]])
                pieces_str = " and ".join(pieces) if pieces else "a casual outfit"
                return {
                    "reply": f"On {log['date']}, you wore your {pieces_str}.",
                    "action": "recent_outfit_log",
                    "data": log,
                }
            return {
                "reply": "You haven't logged any recent outfits yet.",
                "action": "recent_outfit_log",
                "data": None,
            }

        # 4G. Recommended Outfit Today
        if any(p in clean_q for p in ["what should i wear today", "recommend an outfit", "outfit recommendation", "what outfit should i wear", "what to wear"]):
            try:
                rec = OutfitEngine.recommend(user_id)
                items = rec.get("items", [])
                top_item = next((i for i in items if i.get("label") == "Top"), None)
                bot_item = next((i for i in items if i.get("label") == "Bottom"), None)
                if top_item and bot_item:
                    return {
                        "reply": f"Today I recommend your {top_item['name']} paired with {bot_item['name']}.",
                        "action": "recommend_outfit",
                        "data": rec,
                    }
                elif top_item:
                    return {
                        "reply": f"For today, I suggest wearing your {top_item['name']}.",
                        "action": "recommend_outfit",
                        "data": rec,
                    }
                elif rec.get("reason"):
                    return {
                        "reply": f"Outfit suggestion: {rec['reason']}.",
                        "action": "recommend_outfit",
                        "data": rec,
                    }
            except Exception as e:
                logger.warning(f"Error in outfit recommendation: {e}")

        # ======================================================================
        # 5. INDOOR CLIMATE & ESP32 SENSOR
        # ======================================================================
        if any(p in clean_q for p in ["room temperature", "indoor temperature", "how warm is it in here", "temperature inside", "what is the temperature in my room"]):
            if esp32_data and esp32_data.get("temperature") is not None:
                t = float(esp32_data["temperature"])
                return {
                    "reply": f"The room temperature is currently {t:.1f} degrees Celsius.",
                    "action": "room_temperature",
                    "data": esp32_data,
                }
            return {
                "reply": "The indoor room sensor telemetry is currently updating.",
                "action": "room_temperature_unavailable",
            }

        if any(p in clean_q for p in ["room humidity", "indoor humidity", "how humid is it in here", "humidity inside"]):
            if esp32_data and esp32_data.get("humidity") is not None:
                h = int(round(float(esp32_data["humidity"])))
                return {
                    "reply": f"Indoor humidity is currently {h} percent.",
                    "action": "room_humidity",
                    "data": esp32_data,
                }
            return {
                "reply": "Indoor humidity data is currently updating.",
                "action": "room_humidity_unavailable",
            }

        if any(p in clean_q for p in ["motion detected", "is anyone in the room", "motion sensor"]):
            if esp32_data and "motion" in esp32_data:
                m = bool(esp32_data["motion"])
                status = "Motion is currently detected in the room." if m else "No motion detected in the room right now."
                return {
                    "reply": status,
                    "action": "motion_status",
                    "data": {"motion": m},
                }

        if any(p in clean_q for p in ["indoor climate", "room status", "sensor status", "esp32 status"]):
            if esp32_data and esp32_data.get("temperature") is not None:
                t = float(esp32_data["temperature"])
                h = esp32_data.get("humidity")
                h_str = f" with {int(round(float(h)))} percent humidity" if h is not None else ""
                return {
                    "reply": f"The room is currently {t:.1f} degrees Celsius{h_str}.",
                    "action": "indoor_climate",
                    "data": esp32_data,
                }

        # ======================================================================
        # 6. LIVE CAMERA VISION & APPEARANCE
        # ======================================================================
        if any(p in clean_q for p in ["what am i wearing", "what is the camera seeing", "what do you see me wearing", "detect my clothes"]):
            if stylist_data and stylist_data.get("status") == "ok":
                detected = stylist_data.get("detected", {})
                if detected.get("name"):
                    advice = f" Stylist advice: {stylist_data['advice']}" if stylist_data.get("advice") else ""
                    return {
                        "reply": f"The mirror camera detects you are wearing: {detected['name']}.{advice}",
                        "action": "camera_clothing",
                        "data": stylist_data,
                    }
            return {
                "reply": "The smart mirror camera hasn't detected any specific clothing right now.",
                "action": "camera_clothing_none",
            }

        if any(p in clean_q for p in ["how do i look", "how is my hair", "how is my beard", "how is my grooming", "grooming check", "hair check", "appearance check"]):
            if grooming_data and grooming_data.get("status") == "ok":
                hair = grooming_data.get("hair", "")
                beard = grooming_data.get("facial_hair", "")
                skin = grooming_data.get("skin", "")
                advice = grooming_data.get("advice", "")
                parts = []
                if hair:
                    parts.append(f"Hair: {hair}")
                if beard:
                    parts.append(f"facial hair: {beard}")
                if skin:
                    parts.append(f"skin: {skin}")
                summary = "; ".join(parts)
                adv_str = f" {advice}" if advice else ""
                return {
                    "reply": f"Your grooming scan shows: {summary}.{adv_str}",
                    "action": "camera_grooming",
                    "data": grooming_data,
                }
            return {
                "reply": "Live grooming scan is not ready yet. Please stand centered in front of the camera.",
                "action": "camera_grooming_none",
            }

        # ======================================================================
        # 7. OUTDOOR WEATHER & CALENDAR
        # ======================================================================
        if any(p in clean_q for p in ["what is the weather", "how is the weather", "weather today", "will it rain", "forecast today"]):
            try:
                w = WeatherService.get_weather()
                temp = w.get("temperature")
                cond = w.get("condition", "Clear")
                city = w.get("city", "your area")
                high = w.get("high")
                low = w.get("low")
                if temp is not None:
                    h_l = f" High of {high} and low of {low}." if high is not None and low is not None else ""
                    return {
                        "reply": f"Right now in {city}, it's {int(round(float(temp)))} degrees and {cond.lower()}.{h_l}",
                        "action": "weather_query",
                        "data": w,
                    }
            except Exception:
                pass

        if any(p in clean_q for p in ["what is my schedule", "calendar events", "what do i have today", "what meetings do i have", "my agenda"]):
            try:
                events = CalendarService.get_events_for_user(user_id)
                valid = [e for e in events if e.get("title") and "no more upcoming" not in e.get("title", "").lower()]
                if valid:
                    ev_str = "; ".join([f"{e['time']}: {e['title']}" for e in valid])
                    return {
                        "reply": f"On your agenda today: {ev_str}.",
                        "action": "calendar_query",
                        "data": valid,
                    }
                return {
                    "reply": "You have a clear schedule with no upcoming events today.",
                    "action": "calendar_query",
                    "data": [],
                }
            except Exception:
                pass

        # Not handled by fast-path direct action
        return None

    @classmethod
    def build_llm_grounding_prompt(cls, snapshot: Dict[str, Any]) -> str:
        """
        Builds a compact, high-density grounding block for the local Qwen LLM.
        All facts are strictly grounded in real database rows and sensor inputs.
        """
        now = datetime.now()
        user = snapshot.get("active_user", {})
        reminders = snapshot.get("reminders", {}).get("active", [])
        wardrobe = snapshot.get("wardrobe", {})
        categories = wardrobe.get("category_counts", {})
        outfit = snapshot.get("outfit", {})
        rec = outfit.get("recommendation", {})
        history = outfit.get("recent_history", [])
        weather = snapshot.get("weather", {})
        esp32 = snapshot.get("esp32", {})
        vision = snapshot.get("vision", {})
        system = snapshot.get("system", {})
        reg_users = snapshot.get("registered_users", [])
        calendar = snapshot.get("calendar", [])

        facts = [
            f"Current time: {now.strftime('%I:%M %p')}, date: {now.strftime('%A, %B %d, %Y')}.",
            f"Active user: {user.get('name', 'User')} (ID: {user.get('id', 'guest')}). Style preference: {user.get('preferred_style', 'Casual')}. Hair preference: {user.get('hair_preference', 'Textured')}.",
            f"Registered users on mirror: {', '.join(reg_users)}.",
            f"Privacy mode: {'ENABLED' if system.get('privacy_mode') else 'DISABLED'}.",
        ]

        # Reminders
        if reminders:
            rem_str = "; ".join([f"{r['title']} at {r['datetime']}" for r in reminders])
            facts.append(f"Active user's reminders ({len(reminders)}): {rem_str}.")
        else:
            facts.append("Active user's reminders: None.")

        # Wardrobe
        items = wardrobe.get("items", [])
        top_names = [i["name"] for i in items if i.get("category", "").lower() == "top"]
        bot_names = [i["name"] for i in items if i.get("category", "").lower() == "bottom"]
        cats_str = ", ".join([f"{count} {cat}s" for cat, count in categories.items()]) or "0 items"
        facts.append(f"Wardrobe total: {wardrobe.get('total_items', 0)} items ({cats_str}).")
        if top_names:
            facts.append(f"Tops: {', '.join(top_names[:5])}.")
        if bot_names:
            facts.append(f"Bottoms: {', '.join(bot_names[:5])}.")
        if wardrobe.get("most_worn"):
            mw = wardrobe["most_worn"]
            facts.append(f"Most worn item: {mw.get('name')} ({mw.get('times_worn', 0)} times).")

        # Outfit Recommendation & History
        if rec and rec.get("items"):
            rec_pieces = ", ".join([f"{it['label']}: {it['name']}" for it in rec["items"]])
            facts.append(f"Today's recommended outfit: {rec_pieces}.")
        if history:
            h = history[0]
            facts.append(f"Most recent logged outfit on {h.get('date')}.")

        # Weather & Indoor Telemetry
        if weather.get("temperature") is not None:
            facts.append(f"Outdoor weather in {weather.get('city', 'your city')}: {weather.get('temperature')}°C, {weather.get('condition', 'Clear')}. High {weather.get('high')}°C, Low {weather.get('low')}°C.")
        if esp32.get("temperature") is not None:
            facts.append(f"Indoor room temperature: {esp32.get('temperature')}°C, humidity: {esp32.get('humidity', '--')}%.")

        # Schedule
        valid_events = [e for e in calendar if e.get("title") and "no more upcoming" not in e.get("title", "").lower()]
        if valid_events:
            ev_str = "; ".join([f"{e['time']}: {e['title']}" for e in valid_events])
            facts.append(f"Today's schedule: {ev_str}.")

        # Vision
        stylist = vision.get("stylist", {})
        if stylist.get("detected", {}).get("name"):
            facts.append(f"Camera detected user wearing: {stylist['detected']['name']}.")
        groom = vision.get("grooming", {})
        if groom.get("hair") or groom.get("facial_hair"):
            facts.append(f"Camera grooming: Hair: {groom.get('hair')}, Beard: {groom.get('facial_hair')}, Skin: {groom.get('skin')}.")

        facts_block = "\n".join(facts)

        return (
            "You are ReflectAI, an intelligent smart mirror voice assistant connected to all user data and hardware.\n"
            "Keep answers direct, natural, conversational, and under two short sentences. Never use bullet points, markdown, or headers.\n"
            "Answer questions using the exact database and system facts below:\n\n"
            f"{facts_block}"
        )

    @classmethod
    def ask_llm(cls, query: str, snapshot: Dict[str, Any]) -> str:
        """
        Executes query through local LLM with grounded system prompt.
        """
        llm = _get_local_llm()
        if not llm:
            return "I am connected to all your data. Could you rephrase your question?"

        prompt = cls.build_llm_grounding_prompt(snapshot)
        try:
            res = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": query},
                ],
                max_tokens=90,
                temperature=0.6,
            )
            return res["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return "Sorry, I had trouble thinking of a reply right now."

    @classmethod
    def process_query(
        cls,
        query: str,
        user_id: Optional[str] = None,
        esp32_data: Optional[Dict[str, Any]] = None,
        stylist_data: Optional[Dict[str, Any]] = None,
        grooming_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Primary entrypoint for the Voice Assistant:
        1. Evaluates Fast-Path Action & Exact Query Engine.
        2. If matched, performs action/lookup with zero hallucination.
        3. If not matched, pulls full snapshot and queries the grounded LLM.
        """
        if not user_id:
            user_id = SettingsRepository.get("active_user", default="guest")

        # 1. Fast-Path Engine
        action_res = cls.execute_action_or_query(
            query=query,
            user_id=user_id,
            esp32_data=esp32_data,
            stylist_data=stylist_data,
            grooming_data=grooming_data,
        )
        if action_res:
            return {
                "status": "ok",
                "query": query,
                "reply": action_res["reply"],
                "action": action_res["action"],
                "data": action_res.get("data"),
                "source": "action",
                "user_id": user_id,
            }

        # 2. Grounded LLM
        snapshot = cls.get_knowledge_snapshot(
            user_id=user_id,
            esp32_data=esp32_data,
            stylist_data=stylist_data,
            grooming_data=grooming_data,
        )
        llm_reply = cls.ask_llm(query, snapshot)

        return {
            "status": "ok",
            "query": query,
            "reply": llm_reply,
            "action": None,
            "data": None,
            "source": "llm",
            "user_id": user_id,
        }
