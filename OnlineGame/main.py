import json
import random
import socket
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Set, List
from poker import PokerGame

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class GameLogger:
    def __init__(self, filename="game_data.json"):
        self.filename = filename
        if not os.path.exists(self.filename):
            with open(self.filename, "w") as f:
                json.dump([], f)

    def log_game(self, game_entry):
        try:
            with open(self.filename, "r") as f:
                data = json.load(f)
            data.append(game_entry)
            with open(self.filename, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Logging error: {e}")

class GameRoom:
    def __init__(self):
        self.clients: Dict[str, WebSocket] = {}
        self.game_stage = 0  
        self.stages = ["Waiting", "Pre-Flop", "Flop", "Turn", "River", "Showdown"]
        self.current_order = []
        self.history = [] # For UI
        self.poker = None
        self.ready_players: Set[str] = set()
        
        # Data Collection for JSON
        self.player_map = {} # Name -> Index
        self.game_log = {
            "player_count": 0,
            "hole_cards": {},
            "community_cards_at_stages": {},
            "rank_swaps": [],
            "final_rankings_per_turn": []
        }

room = GameRoom()
logger = GameLogger()

@app.on_event("startup")
async def startup_event():
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"\n🚀 SERVER LIVE: http://{local_ip}:8000\n")

@app.get("/")
async def get():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

async def broadcast_state():
    for name, ws in list(room.clients.items()):
        try:
            state = {
                "type": "state",
                "stage": room.stages[room.game_stage],
                "players_connected": list(room.clients.keys()),
                "current_order": room.current_order,
                "history": room.history,
                "community_cards": [str(c) for c in room.poker.community_cards] if room.poker else [],
                "my_hand": [str(c) for c in room.poker.players.get(name, [])] if room.poker else [],
                "ready_count": len(room.ready_players),
                "is_ready": name in room.ready_players,
                "game_over": room.game_stage == 5
            }
            if room.game_stage == 5:
                matches, actual_order = room.poker.compare_players(room.current_order)
                state["results"] = {"won": matches, "actual_order": actual_order}
            await ws.send_text(json.dumps(state))
        except: pass

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    name = None
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            action = payload.get("action")

            if action == "join":
                name = payload.get("name")
                room.clients[name] = websocket
                await broadcast_state()

            elif action in ["start", "reset"]:
                if len(room.clients) >= 3:
                    player_names = list(room.clients.keys())
                    # Reset Data Collection
                    room.player_map = {name: idx for idx, name in enumerate(player_names)}
                    room.game_log = {
                        "player_indices": list(room.player_map.values()),
                        "hole_cards": {},
                        "swaps": [],
                        "turn_rankings": []
                    }
                    
                    # Init Game
                    room.poker = PokerGame(player_names)
                    for p in player_names:
                        cards = room.poker.deck.deal(2)
                        room.poker.players[p] = cards
                        room.game_log["hole_cards"][room.player_map[p]] = [str(c) for c in cards]
                    
                    room.current_order = player_names.copy()
                    random.shuffle(room.current_order)
                    room.game_stage = 1
                    room.history = []
                    room.ready_players = set()
                    await broadcast_state()

            elif action == "swap":
                target_rank = int(payload.get("rank")) - 1
                if 0 <= target_rank < len(room.current_order):
                    old_order = [room.player_map[p] for p in room.current_order]
                    idx = room.current_order.index(name)
                    # Perform Swap
                    room.current_order[idx], room.current_order[target_rank] = \
                        room.current_order[target_rank], room.current_order[idx]
                    
                    # Log Swap
                    new_order = [room.player_map[p] for p in room.current_order]
                    room.game_log["swaps"].append({
                        "player_index": room.player_map[name],
                        "from_order": old_order,
                        "to_order": new_order,
                        "stage": room.stages[room.game_stage]
                    })
                    await broadcast_state()

            elif action == "toggle_ready":
                if name in room.ready_players: room.ready_players.remove(name)
                else: room.ready_players.add(name)
                
                if len(room.ready_players) == len(room.clients):
                    # Record turn ranking
                    room.game_log["turn_rankings"].append({
                        "stage": room.stages[room.game_stage],
                        "ranking": [room.player_map[p] for p in room.current_order],
                        "community_cards": [str(c) for c in room.poker.community_cards]
                    })
                    
                    room.history.append({"stage": room.stages[room.game_stage], "order": list(room.current_order)})
                    if room.game_stage == 1: room.poker.community_cards.extend(room.poker.deck.deal(3))
                    elif room.game_stage in [2, 3]: room.poker.community_cards.extend(room.poker.deck.deal(1))
                    
                    room.game_stage += 1
                    room.ready_players = set()
                    
                    if room.game_stage == 5: # Game Over
                        matches, actual_order = room.poker.compare_players(room.current_order)
                        room.game_log["outcome"] = {
                            "success": matches,
                            "actual_order": [room.player_map[p] for p in actual_order]
                        }
                        logger.log_game(room.game_log)
                
                await broadcast_state()

    except WebSocketDisconnect:
        if name in room.clients: del room.clients[name]
        await broadcast_state()