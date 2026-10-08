# Devcontainer: connecting to the robot over the network

This devcontainer uses Docker Desktop's default bridge networking, unchanged from the original
working setup (`6080:80` for the noVNC desktop). ROS2 DDS discovery relies on UDP multicast by
default, which may or may not traverse Docker Desktop for macOS's VM/NAT boundary to reach the
RubikPi on the same WiFi network — this hasn't been validated yet.

## First, just try it

With the Mac and the RubikPi on the same WiFi network, open a terminal in the container and run:

```bash
ros2 topic list
```

If the robot's topics (e.g. `motor_commands`, `imu_data`) show up, multicast discovery is working
and no further changes are needed.

## If discovery doesn't work: switch to CycloneDDS with an explicit peer

Multicast often doesn't survive container NAT, but direct unicast between two known IPs usually
does. To force that:

1. Find the RubikPi's LAN IP (e.g. `ping rubikpi.local` or check its network settings).
2. Add to the `ros2` service in `docker-compose.yml`:
   ```yaml
   environment:
     RMW_IMPLEMENTATION: rmw_cyclonedds_cpp
     CYCLONEDDS_URI: "<CycloneDDS><Domain><Peers><Peer address=\"ROBOT_IP_HERE\"/></Peers></Domain></CycloneDDS>"
   ```
   replacing `ROBOT_IP_HERE` with the actual IP from step 1.
3. Rebuild/restart the container and re-check `ros2 topic list`.

Don't add this config speculatively — only once plain multicast is confirmed not to work, so you're
not debugging two unknowns at once.
