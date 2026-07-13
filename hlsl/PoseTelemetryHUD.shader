Shader "Custom/PoseTelemetryHUD"
{
    Properties
    {
        _BlockPx ("Block Size (px)", Float) = 4
        _OffsetX ("Offset X from left (px)", Float) = 8
        _OffsetY ("Offset Y from top (px)", Float) = 8
    }
    SubShader
    {
        Tags { "Queue"="Overlay+1000" "RenderType"="Overlay" "IgnoreProjector"="True" }
        ZTest Always
        ZWrite Off
        Cull Off
        Blend Off

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 5.0
            #include "UnityCG.cginc"

            float _BlockPx;
            float _OffsetX;
            float _OffsetY;

            float _VRChatCameraMode;
            float _VRChatMirrorMode;
            uint  _VRChatTimeNetworkMs;

            struct appdata { float4 vertex : POSITION; float2 uv : TEXCOORD0; };
            struct v2f     { float4 pos : SV_Position; };

            v2f vert (appdata v)
            {
                v2f o;
                if (_VRChatCameraMode != 0.0 || _VRChatMirrorMode != 0.0)
                {
                    o.pos = float4(2e6, 2e6, 2e6, 1.0);
                    return o;
                }
                float2 ndc = v.uv * 2.0 - 1.0;
                o.pos = float4(ndc, UNITY_NEAR_CLIP_VALUE, 1.0);
                return o;
            }

            #define ROWS 12
            #define COLS 32
            static const uint MAGIC = 0x5AC3E7A1u;

            fixed4 frag (v2f i) : SV_Target
            {
                float2 p = i.pos.xy;
                if (_ProjectionParams.x < 0.0)
                    p.y = _ScreenParams.y - p.y;
                p -= float2(_OffsetX, _OffsetY);
                float2 gridPx = float2(COLS, ROWS) * _BlockPx;

                if (p.x < 0.0 || p.y < 0.0 || p.x >= gridPx.x || p.y >= gridPx.y)
                    clip(-1.0);

                uint col = (uint)(p.x / _BlockPx);
                uint row = (uint)(p.y / _BlockPx);

                float3 camPos = _WorldSpaceCameraPos;
                float3 fwd = -UNITY_MATRIX_V[2].xyz;
                float3 up  =  UNITY_MATRIX_V[1].xyz;

                uint w[ROWS];
                w[0]  = MAGIC;
                w[1]  = _VRChatTimeNetworkMs;
                w[2]  = asuint(camPos.x);
                w[3]  = asuint(camPos.y);
                w[4]  = asuint(camPos.z);
                w[5]  = asuint(fwd.x);
                w[6]  = asuint(fwd.y);
                w[7]  = asuint(fwd.z);
                w[8]  = asuint(up.x);
                w[9]  = asuint(up.y);
                w[10] = asuint(up.z);
                w[11] = w[0]^w[1]^w[2]^w[3]^w[4]^w[5]
                      ^ w[6]^w[7]^w[8]^w[9]^w[10];

                uint bit = (w[row] >> (31u - col)) & 1u;
                return bit ? fixed4(1,1,1,1) : fixed4(0,0,0,1);
            }
            ENDCG
        }
    }
    Fallback Off
}
