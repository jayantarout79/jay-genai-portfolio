"use client";

import { useState } from "react";

const STORAGE_KEY = "ingrediscore_device_id";

export function useDeviceId() {
  const [deviceId, setDeviceId] = useState<string | null>(() => {
    if (typeof window === "undefined") return null;
    const existing = window.localStorage.getItem(STORAGE_KEY);
    if (existing) return existing;
    const nextId = crypto.randomUUID();
    window.localStorage.setItem(STORAGE_KEY, nextId);
    return nextId;
  });
  const [ready] = useState(() => typeof window !== "undefined");

  return { deviceId, ready, setDeviceId };
}
