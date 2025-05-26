export default function EnvTest() {
  return <div style={{fontSize:12}}>API: {process.env.NEXT_PUBLIC_API_URL}</div>;
}
