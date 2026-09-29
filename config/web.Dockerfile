ARG NODE_IMAGE
ARG PYTHON_IMAGE
FROM ${NODE_IMAGE} AS frontend
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM ${PYTHON_IMAGE}
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY config/web-requirements.txt /app/config/web-requirements.txt
RUN pip install --no-cache-dir -r config/web-requirements.txt
COPY src/__init__.py src/image_web.py src/image_history.py src/image_edit.py src/offline.py /app/src/
COPY --from=frontend /web/dist /app/web/dist
COPY web/LICENSE.shadcn /app/licenses/shadcn-ui.LICENSE
USER 1000:1000
