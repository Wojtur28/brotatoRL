extends "res://entities/units/movement_behaviors/player_movement_behavior.gd"

# While the RL bridge controls the run, the player moves in the direction of the agent's
# last action instead of reading the keyboard or gamepad.

const BRIDGE_META := "brotato_rl_bridge"


func get_movement() -> Vector2:
	if Engine.has_meta(BRIDGE_META):
		var bridge = Engine.get_meta(BRIDGE_META)
		if bridge.controls_player:
			return bridge.move
	return .get_movement()
