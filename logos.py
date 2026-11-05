"""The app's logo, and the action's logo.

The app's logo is a LOGO blok (an ``<AgentLogo>`` root), which a client shows beside the
app's name. The action's logo is an ACTION_LOGO blok (an ``<ActionLogo>`` root), painted
only with the viewer's pens (``ink``, ``muted``, ``brand``, ``brand-strong``,
``brand-soft``, ``surface``), so it matches every other app's actions and follows the
viewer's brand and dark mode. A shape that ``animate``s plays while the logo is hovered,
and while the viewer's run of the action is going.

Blok names are the organization's, so each carries the app's name.
"""

from arkitekt import App

APP = "ais_organelles_app"

APP_LOGO = f"{APP}_logo"

# A neuron: its soma, and the axon leaving it. The axon's first stretch, the initial
# segment, is a tube whose wall glows with AnkG (green), drawn as the ring-like stain it is;
# inside it lie three cisternal organelles (synaptopodin, red), off the axis and in path
# order, on the drawn path, which runs dashed through the segment.
APP_LOGO_SOURCE = """
<AgentLogo viewBox="0 0 64 64" title="ais-organelles-app">
  <Rect x="0" y="0" width="64" height="64" rx="14" fill="ink" />
  <G stroke="muted" strokeLinecap="round" fill="none">
    <Path d="M12 41 Q9 35 4.5 32.5" strokeWidth="3" />
    <Path d="M8.5 37 L5.5 39" strokeWidth="2" />
    <Path d="M10 51 Q6 53 4 57.5" strokeWidth="3" />
    <Path d="M19.5 55.5 Q21 58.5 24.5 60" strokeWidth="3" />
  </G>
  <Path d="M43.5 22 C45.7 16.4 50 10 58.5 7.5" fill="none" stroke="muted" strokeWidth="3" strokeLinecap="round" />
  <Circle cx="16.5" cy="47" r="9.5" fill="muted" />
  <Path d="M24.5 40.5 C27 31 38 36 43.5 22" fill="none" stroke="spectrum-green" strokeWidth="12.5" strokeLinecap="round" />
  <Path d="M24.5 40.5 C27 31 38 36 43.5 22" fill="none" stroke="ink" strokeWidth="7.5" strokeLinecap="round" />
  <Path d="M24.5 40.5 C27 31 38 36 43.5 22" fill="none" stroke="surface" strokeWidth="1" strokeDasharray="1.6 2" strokeLinecap="round" strokeOpacity="0.6" animate="march" />
  <G fill="spectrum-red">
    <Ellipse cx="27.9" cy="37.3" rx="2.6" ry="1.35" transform="rotate(-43 27.9 37.3)" animate="pulse" />
    <Ellipse cx="32.3" cy="31.7" rx="2.6" ry="1.35" transform="rotate(-24 32.3 31.7)" animate="pulse" animateDelay="0.2" />
    <Ellipse cx="40.7" cy="29.3" rx="2.6" ry="1.35" transform="rotate(-47 40.7 29.3)" animate="pulse" animateDelay="0.4" />
  </G>
  <Rect x="0.5" y="0.5" width="63" height="63" rx="13.5" fill="none" stroke="muted" strokeOpacity="0.3" strokeWidth="1" />
</AgentLogo>
"""

ACTION_LOGOS: dict[str, str] = {
    # A tube round a dashed path, three organelles inside it, in path order.
    "segment_ais_tube_along_path": """
<ActionLogo viewBox="0 0 24 24" title="Segment AIS Tube Along Path">
  <Path d="M4 19 Q11 13 19.5 5" fill="none" stroke="ink" strokeWidth="7.5" strokeLinecap="round" />
  <Path d="M4 19 Q11 13 19.5 5" fill="none" stroke="surface" strokeWidth="4.5" strokeLinecap="round" />
  <Path d="M4 19 Q11 13 19.5 5" fill="none" stroke="brand" strokeWidth="1" strokeDasharray="1.4 1.4" strokeLinecap="round" animate="march" />
  <G fill="brand-strong">
    <Circle cx="7.3" cy="17" r="1.2" animate="pulse" />
    <Circle cx="11" cy="12" r="1.2" animate="pulse" animateDelay="0.2" />
    <Circle cx="16.6" cy="8.6" r="1.2" animate="pulse" animateDelay="0.4" />
  </G>
</ActionLogo>
""",
}


def logo_of(action: str) -> str:
    """The name of the action logo blok that draws ``action`` (its function's name)."""
    if action not in ACTION_LOGOS:
        raise KeyError(f"No logo is drawn for the action {action!r}")
    return f"{APP}_{action}"


def declare_logos(app: App) -> None:
    """Declares the app's logo, as its default, and every action logo."""
    app.blok(
        APP_LOGO,
        APP_LOGO_SOURCE,
        description="A neuron whose axon initial segment holds three cisternal organelles, along a drawn path.",
        default=True,
    )
    for action, source in ACTION_LOGOS.items():
        app.blok(logo_of(action), source)
