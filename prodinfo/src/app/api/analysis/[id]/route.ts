import { NextRequest, NextResponse } from "next/server";
import { supabaseServerClient } from "@/lib/supabase";

export async function GET(
  req: NextRequest,
  { params }: { params: { id: string } },
) {
  if (!supabaseServerClient) return NextResponse.json({ error: "Server not configured" }, { status: 500 });
  const deviceId = req.nextUrl.searchParams.get("device_id");
  if (!deviceId) return NextResponse.json({ error: "device_id required" }, { status: 400 });

  const { data, error } = await supabaseServerClient
    .from("product_analyses")
    .select("*")
    .eq("id", params.id)
    .eq("device_id", deviceId)
    .single();

  if (error) {
    const status = error.code === "PGRST116" ? 404 : 500;
    return NextResponse.json({ error: "Not found" }, { status });
  }

  return NextResponse.json(data);
}

export async function DELETE(
  req: NextRequest,
  { params }: { params: { id: string } },
) {
  if (!supabaseServerClient) return NextResponse.json({ error: "Server not configured" }, { status: 500 });
  const deviceId = req.nextUrl.searchParams.get("device_id");
  if (!deviceId) return NextResponse.json({ error: "device_id required" }, { status: 400 });

  const { error } = await supabaseServerClient
    .from("product_analyses")
    .delete()
    .eq("id", params.id)
    .eq("device_id", deviceId);

  if (error) {
    console.error(error);
    return NextResponse.json({ error: "Delete failed" }, { status: 500 });
  }

  return NextResponse.json({ status: "deleted" });
}
