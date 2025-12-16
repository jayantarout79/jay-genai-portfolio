-- Create storage bucket
insert into storage.buckets (id, name, public)
values ('product-labels', 'product-labels', true)
on conflict (id) do update set public = true;

-- Table for analyses
create extension if not exists "uuid-ossp";
create extension if not exists pgcrypto;

create table if not exists public.product_analyses (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  device_id text not null,
  image_urls jsonb not null,
  raw_extraction jsonb not null,
  analysis_result jsonb not null,
  product_name text,
  brand text,
  category text,
  buy_score_percent int,
  verdict text,
  confidence numeric
);

create index if not exists idx_product_analyses_created_at on public.product_analyses (created_at desc);
create index if not exists idx_product_analyses_device_id on public.product_analyses (device_id);
