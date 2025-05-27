import React from "react";

export function DevModeBanner() {
  if (process.env.NODE_ENV !== "development") return null;
  return (
    <span style={{
      color: "red",
      fontWeight: "bold",
      fontSize: 16,
      marginLeft: 12,
      letterSpacing: 1,
      textShadow: "0 1px 2px #fff"
    }}>
      DEV MODE
    </span>
  );
}
