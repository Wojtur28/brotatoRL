extends "res://singletons/announcement_manager.gd"

# Keep the game offline: the parent _ready() always fires its HTTP request (Godot 3 calls
# _ready on every inheritance level), so point it at a closed local port instead.


func _init() -> void:
	_announcement_url = "http://127.0.0.1:9/"
	_asset_url = "http://127.0.0.1:9/"
