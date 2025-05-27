import React from "react";

export const AnimatedEllipsis: React.FC = () => (
  <span style={{ display: 'inline-block', width: 24 }}>
    <span className="animate-pulse">...</span>
  </span>
);
