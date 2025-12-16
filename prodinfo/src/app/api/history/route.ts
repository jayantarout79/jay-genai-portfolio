import { NextRequest, NextResponse } from "next/server";
import { supabaseServerClient } from "@/lib/supabase";

export async function GET(req: NextRequest) {
  if (!supabaseServerClient) return NextResponse.json({ error: "Server not configured" }, { status: 500 });

  const deviceId = req.nextUrl.searchParams.get("device_id");
  if (!deviceId) return NextResponse.json({ error: "device_id required" }, { status: 400 });

  const { data, error } = await supabaseServerClient
    .from("product_analyses")
    .select("id, created_at, product_name, brand, buy_score_percent, verdict, category")
    .eq("device_id", deviceId)
    .order("created_at", { ascending: false })
    .limit(50);

  if (error) {
    console.error(error);
    return NextResponse.json({ error: "Failed to fetch" }, { status: 500 });
  }

  return NextResponse.json(data);
}

export async function DELETE(req: NextRequest) {
  if (!supabaseServerClient) return NextResponse.json({ error: "Server not configured" }, { status: 500 });
  const deviceId = req.nextUrl.searchParams.get("device_id");
  if (!deviceId) return NextResponse.json({ error: "device_id required" }, { status: 400 });

  const { error } = await supabaseServerClient
    .from("product_analyses")
    .delete()
    .eq("device_id", deviceId);

  if (error) {
    console.error(error);
    return NextResponse.json({ error: "Failed to delete" }, { status: 500 });
  }

  return NextResponse.json({ status: "ok" });
}
